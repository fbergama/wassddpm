###################################################################################
# WASSDDPM: Denoising Diffusion Probabilistic Models for scattered point cloud    #
#           interpolation of sea waves elevation data                             #
# Copyright (C) 2026 Ca' Foscari University of Venice                             #
#                                                                                 #
# This program is free software: you can redistribute it and/or modify it under   #
# the terms of the GNU General Public License as published by the Free Software   #
# Foundation, either version 3 of the License, or (at your option) any later      #
# version.                                                                        #
#                                                                                 #
# This program is distributed in the hope that it will be useful, but WITHOUT ANY #
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A #
# PARTICULAR PURPOSE. See the GNU General Public License for more details.        #
#                                                                                 #
# You should have received a copy of the GNU General Public License along         #
# with this program. If not, see <https://www.gnu.org/licenses/>.                 #
###################################################################################
#
#  Author(s):
#  - Filippo Bergamasco 
#
###################################################################################

import importlib.metadata
import argparse
import sys
import numpy as np
import netCDF4
import h5py
import cv2 as cv
import matplotlib.pyplot as plt
import tqdm
import cmocean
from .conditional_ddim_interpolate import conditional_DDIM_interpolate


def generate_dataset_from_wassfast( wassfast_file, out_h5_file ):
    outvideo = None

    print(f"Opening {wassfast_file} for reading")
    with netCDF4.Dataset( wassfast_file ) as ds_wassfast:

        N = ds_wassfast["/Z"].shape[0]
        zmin = ds_wassfast["/meta"].getncattr("zmin").item(0)
        zmax = ds_wassfast["/meta"].getncattr("zmax").item(0)
        zmean = ds_wassfast["/meta"].getncattr("zmean").item(0)
        maxzv = max( np.abs(zmin), np.abs(zmax) )

        fps = int( 1.0 / (ds_wassfast["/time"][1] - ds_wassfast["/time"][0]) )

        print("Sequence info: ")
        print(f"{N} frames, zmin={zmin}, zmax={zmax}, fps={fps}")
        
        print(f"Opening {out_h5_file} for writing")
        with h5py.File( out_h5_file, "w") as out:

            
            GPM = out.create_dataset("GPM", (N, ds_wassfast["/Z"].shape[1], ds_wassfast["/Z"].shape[2], 3), dtype='f4')
            wassfast = out.create_dataset("WASSfast", (N, ds_wassfast["/Z"].shape[1], ds_wassfast["/Z"].shape[2]), dtype='f4')
            t = out.create_dataset("t", (N,), dtype='f4')

            
    
            zmin=-maxzv
            zmax=maxzv
            zmean=0.0

            GPM.attrs['zminmax'] = (zmin,zmax)
            GPM.attrs['zmean'] = zmean
            GPM.attrs['du'] = ((ds_wassfast["/X_grid"][0,1]).item(0) - (ds_wassfast["/X_grid"][0,0]).item(0) ) / 1000.0
            GPM.attrs['dv'] = ((ds_wassfast["/Y_grid"][1,0]).item(0) - (ds_wassfast["/Y_grid"][1,0]).item(0) ) / 1000.0

            
            for idx in tqdm.trange( N ):
            
                Zwassfast = ds_wassfast["/Z"][idx,:,:] / 1000.0
                
                Pts = ds_wassfast["/Zinput"][idx,:,:]
                mask = ( np.logical_not( np.isnan(Pts) ) ).astype(np.float32)
                Pts[mask==0]=0.0

                GT = np.copy(Zwassfast).astype(np.float32)
    
                GPM[idx,:,:,0] = Zwassfast  # Used for Kalman, and for comparisons later on. Not used for interpolation
                GPM[idx,:,:,1] = Pts.astype(np.float32)
                GPM[idx,:,:,2] = mask.astype(np.float32)

                t[idx] = ds_wassfast["/time"][idx]

                wassfast[idx,:,:] = Zwassfast
                

                plt.figure( figsize=(15,4) )
                plt.subplot(1,2,1)
                plt.imshow(Zwassfast, vmin=zmin, vmax=zmax )
                plt.colorbar()
                plt.title("WASSfast")
                plt.subplot(1,2,2)
                plt.imshow(ds_wassfast["/Zinput"][idx,:,:], vmin=zmin, vmax=zmax )
                plt.title("WASSfast points")
                plt.colorbar()
                plt.tight_layout()
    
                plt.savefig( "/tmp/aux.jpg" )
                plt.close()
                aux = cv.imread( "/tmp/aux.jpg" )

                if outvideo is None:
                    fourcc = cv.VideoWriter_fourcc(*'mp4v')
                    outvideo = cv.VideoWriter('waves_sequence.mp4', fourcc, fps, (aux.shape[1],  aux.shape[0]))
                
                outvideo.write(aux)
                

    if not outvideo is None:    
        outvideo.release()



def compute_phase_diff_matrix( KX_ab, KY_ab, xsign, ysign, dt, depth=np.inf, current_vector=[0.0, 0.0] ):
    'computes complex matrix for current scene'
    
    #xsign = -np.sign(np.cos(angle))
    #ysign = -np.sign(np.sin(angle))
    #xsign = -xsign
    #ysign = -ysign
    
    Kmag = np.sqrt( KX_ab*KX_ab + KY_ab*KY_ab )
    Ksign = np.sign( (xsign*KX_ab) + ( ysign*KY_ab) )

    omega_sq = 9.8 * Kmag * ( 1.0 if depth == np.inf else np.tanh( Kmag*depth )  )

    ph_diff = Ksign*( np.sqrt(omega_sq) + KX_ab*current_vector[0] + KY_ab*current_vector[1] )*dt
    ph_diff = np.triu(ph_diff) - np.tril(ph_diff )

    return np.exp(1j*np.fft.fftshift( ph_diff*( np.triu(-np.ones(KX_ab.shape)) + np.tril(+np.ones(KX_ab.shape)) ) ) )



def spec_predict( Z, phdiff ):
    l_spec = np.fft.fft2( Z )
    l_spec =  l_spec * phdiff
    Z_predict = np.real(np.fft.ifft2(l_spec))  
    return Z_predict



def postproc_Kalman( h5file, gen_dir, out_ncfile, KALMAN_PROCESS_STD=0.04, do_plots=False ):

    print(f"Opening {out_ncfile} for writing..")
    with netCDF4.Dataset( out_ncfile, "r+" ) as outf:

        with h5py.File( h5file, 'r') as f:
            data = f['/GPM']
            N = data.shape[0]
            print(N, "frames")

            d = {i:data.attrs[i] for i in list(data.attrs)}
            d["t"] = np.array(f['t'])
            zmin, zmax = data.attrs["zminmax"]

            print(f"data range: {zmin} .. {zmax}")

            H = data.shape[1]
            W = data.shape[2]

            xx, yy = np.meshgrid(np.arange(0,W, dtype=float), np.arange(0,H, dtype=float))
            xx *= d['du'].item(0)
            yy *= d['du'].item(0)

            x_spacing = xx[0,1]-xx[0,0]
            y_spacing = yy[1,0]-yy[0,0]

            print(f"grid x spacing: {x_spacing}")
            print(f"grid y spacing: {y_spacing}")

            assert( abs(x_spacing - y_spacing) < 1E-2 )

            Nmx = W//2
            Nmy = H//2

            kx_ab = np.array( [float(i)/W*(2.0*np.pi/x_spacing)  for i in range(-Nmx,Nmx)] )
            ky_ab = np.array( [float(i)/H*(2*np.pi/y_spacing)  for i in range(-Nmy,Nmy)] )
            KX_ab, KY_ab = np.meshgrid( kx_ab, ky_ab)
            spec_scale = 1.0/(W*H)

            if not "Z_wassfast" in outf.variables:
                print("Renaming /Z to /Z_wassfast in nc file")
                outf.renameVariable("Z","Z_wassfast")

            if not "Z" in outf.variables:
                outf.createVariable("Z", outf["/Z_wassfast"].datatype,outf["/Z_wassfast"].dimensions )


            print("Applying Kalman filter to the interpolated surface samples...")

            idx = 0
            prev_Z = (data[idx,:,:,0] - zmin) / (zmax-zmin)
            prev_Z[np.isnan(prev_Z)] = 0
            prev_t = d["t"][ idx ].item(0)

            outf["/Z"][idx,:,:] = ( prev_Z * (zmax-zmin) - zmin )*1000

            for idx in tqdm.trange(1,N):
                GT = (data[idx,:,:,0] - zmin) / (zmax-zmin)
                PTS = (data[idx,:,:,1] - zmin) / (zmax-zmin)
                MASK = data[idx,:,:,2].astype(np.uint8)
                PTS[ MASK==0 ] = np.nan
                
                filename = "%s/gen_image_%06d.h5"%(gen_dir,idx)

                try:
                    with h5py.File(filename, 'r') as f:

                        gen_surf = f['gen_wave']
                        gen_std = np.std(gen_surf, axis=0)

                        #per_px_mae = np.mean(np.abs(gen_surf - GT), axis=0)

                        # predict from prev surface
                        t = d["t"][ idx ].item(0)
                        dt = t - prev_t
                        
                        phdiff = compute_phase_diff_matrix( KX_ab, KY_ab, -1, 1, dt )
                        Z_predict = spec_predict( prev_Z, phdiff )

                        # select closest to prediction
                        # dist = np.mean(np.abs(gen_surf[:,50:200,50:200] - Z_predict[50:200,50:200]), axis=(1,2))
                        # min_idx = np.argmin(dist)
                        # Z_pred_best_surf = gen_surf[min_idx,...]    

                        # select closest to the mean
                        avg_surf = np.mean(gen_surf[:,50:200,50:200], axis=0)
                        dist = np.mean(np.square(gen_surf[:,50:200,50:200] - avg_surf), axis=(1,2))
                        min_idx = np.argmin(dist)
                        Z_avg_best_surf = gen_surf[min_idx,...]

                        # print("Prediction RMS = ", np.sqrt(np.mean(np.square(GT-Z_predict))))
                        # print("Z best pred-closest RMS = ", np.sqrt(np.mean(np.square(GT-Z_best_surf))))
                        # print("Z best avg-closest RMS = ", np.sqrt(np.mean(np.square(GT-Z_avg_best_surf))))

                        del min_idx

                        #select the best surface to compute Z_final
                        Z_best_surf = Z_avg_best_surf
                        #Z_best_surf = Z_pred_best_surf

                        Qn = np.ones_like( gen_std ) * KALMAN_PROCESS_STD 
                        Rn = gen_std
                        K = np.square(Qn)/(np.square(Qn)+np.square(Rn))
                        
                        Z_final = Z_predict + K*(Z_best_surf-Z_predict)
                        #print("Z final RMS = ", np.sqrt(np.mean(np.square(GT-Z_final))))

                        prev_t = t
                        prev_Z = Z_final
                        
                        outf["/Z"][idx,:,:] = ( prev_Z*(zmax-zmin) + zmin ) * 1000.0

                        if do_plots:
                            plt.figure( figsize=(15,10))
                            plt.subplot(1,3,1)
                            plt.imshow(outf["/Z_wassfast"][idx,...], cmap=cmocean.cm.deep_r, vmin=zmin*1000, vmax=zmax*1000 )
                            plt.title("WASSfast idx=%06d"%idx)

                            plt.subplot(1,3,2)
                            MASKPT = np.clip( np.dstack((MASK,MASK,MASK)).astype(np.uint8)*255, 0, 255)
                            plt.imshow(MASKPT)
                            plt.title("Points")

                            plt.subplot(1,3,3)

                            plt.imshow(outf["/Z"][idx,...], cmap=cmocean.cm.deep_r, vmin=zmin*1000, vmax=zmax*1000 )
                            plt.title("DDPM+KF")
                            plt.savefig("%s/%04d.png"%("plots",idx), bbox_inches='tight')
                            plt.close()


                except FileNotFoundError:
                    print(f"{filename} not found. Aborting.")
                    break



def do_main():

    version = importlib.metadata.version("wassddpm")

    parser = argparse.ArgumentParser(
        description="WASSDDPM %s - Denoising Diffusion Probabilistic Models for scattered point cloud interpolation of sea waves elevation data."%version,
        prog="wassddpm"
    )

    # Global options (can be applied to any action)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--do_plots", 
        action="store_true", 
        help="Enable plotting of results."
    )

    # subparsers for the 'action' argument
    subparsers = parser.add_subparsers(dest="action", required=True, help="The action to perform")

    # --- Action: createdataset ---
    # Format: [infile: NetCDF] [outfile: HDF5]
    parser_create = subparsers.add_parser(
        "createdataset", 
        parents=[parent_parser],
        help="Convert NetCDF to HDF5 dataset"
    )
    parser_create.add_argument("infile", help="Input NetCDF file")
    parser_create.add_argument("outfile", help="Output HDF5 file")

    # --- Action: ddpm ---
    # Format: [infile: HDF5] [outfile: Directory]
    parser_ddpm = subparsers.add_parser(
        "ddpm", 
        parents=[parent_parser],
        help="Run DDPM on HDF5 file"
    )
    parser_ddpm.add_argument("infile", help="Input HDF5 file (from createdataset)")
    parser_ddpm.add_argument("outdir", help="Output directory for temporary results")
    # Specific option for 'ddpm'
    parser_ddpm.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Model inference batch size"
    )

    # --- Action: kf ---
    # Format: [infile: HDF5] [outfile: Directory] [netcdf_to_append]
    parser_kf = subparsers.add_parser(
        "kf", 
        parents=[parent_parser],
        help="Run Kalman Filter"
    )
    parser_kf.add_argument("infile", help="Input HDF5 file")
    parser_kf.add_argument("outfile", help="Directory containing DDPM results")
    parser_kf.add_argument("netcdf_append", help="Initial NetCDF file to append data to")
    
    # Specific option for 'kf'
    parser_kf.add_argument(
        "--kalman_std", 
        type=float, 
        default=0.04, 
        help="Standard deviation for Kalman Filter (default: 0.04)"
    )

    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()



    print(f"--- Running Action: {args.action} ---")
    
    if args.do_plots:
        print(">> Plots enabled")

    if args.action == "createdataset":
        print(f"Reading NetCDF: {args.infile}")
        print(f"Saving HDF5 to: {args.outfile}")
        generate_dataset_from_wassfast( wassfast_file=args.infile, out_h5_file=args.outfile )

    elif args.action == "ddpm":
        print(f"Reading HDF5: {args.infile}")
        print(f"Saving results to directory: {args.outdir}")
        with h5py.File( args.infile, "r") as h5f:
            conditional_DDIM_interpolate( h5f, args.outdir, BATCH_SIZE=args.batch_size )

    elif args.action == "kf":
        print(f"Reading HDF5: {args.infile}")
        print(f"Reading Results Dir: {args.outfile}")
        print(f"Appending to NetCDF: {args.netcdf_append}")
        print(f"Kalman Std: {args.kalman_std}")
        postproc_Kalman( args.infile, args.outfile, args.netcdf_append, args.kalman_std, args.do_plots )




if __name__ == "__main__":
    do_main()

