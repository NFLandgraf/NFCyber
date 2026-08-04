from pathlib import Path
import isx
import tifffile

'''
This code takes the raw .isxd files and does the following:
1. Preprocess (Temporal downsampling: Binning via averaging, Spatial: Binning via averaging)
2. Spatial Bandpass
3. Motion Correction
4. Df/f
5. maximum Projection of df/f
6. CNMFe
'''

import_folder = Path("/Users/lpaeger/Desktop/Miniscope Recordings/MPOA_Nana")
export_folder = Path("/Users/lpaeger/Desktop/Miniscope Recordings/MPOA_Nana/out")
tempor_folder = Path("/Users/lpaeger/Desktop/Miniscope Recordings/MPOA_Nana/temp")
export_folder.mkdir(parents=True, exist_ok=True)
tempor_folder.mkdir(parents=True, exist_ok=True)

rec_files = sorted([p for p in import_folder.iterdir() if ".isxd" in p.name])
print(f"Found {len(rec_files)} .isxd files in {import_folder}")

def get_shape(path):
    with tifffile.TiffFile(path) as tif:
        return tif.asarray().shape

def get_crop(base):
    if '32' in base:
        crop = [110, 50, 850, 700]
    elif '43' in base:
        crop = [130, 120, 750, 680]
    elif '48' in base:
        crop = [300, 50, 800, 670]
    elif '51' in base:
        crop = [250, 50, 750, 650]
    elif '56' in base:
        crop = [70, 80, 880, 720]
    elif '66' in base:
        crop = [250, 150, 850, 650]
    elif '72' in base:
        crop = [250, 200, 700, 600]
    elif '77' in base:
        crop = [350, 150, 750, 650]

    return crop


def preprocess(file_in, base, tempor_folder, export_file, export=True):

    # Preprocess
    print('PP')
    to_crop = get_crop(base)
    file_out = isx.make_output_file_path(file_in, str(tempor_folder), 'PP')
    isx.preprocess(file_in, file_out, 
                temporal_downsample_factor=2, 
                spatial_downsample_factor=2,
                crop_rect=to_crop, #[top_left_x, top_left_y, width, height] when tlwh
                crop_rect_format = "tlwh", 
                fix_defective_pixels=True, 
                trim_early_frames=False)
    
    print('PP done')
    return file_out

def spatial_bandpass(file_in, base, tempor_folder, export_file, export=True):

    # Spatial filter (bandpass)
    print('BP')
    file_out = isx.make_output_file_path(file_in, str(tempor_folder), 'BP')
    isx.spatial_filter(file_in, file_out, 
                    low_cutoff=0.005, 
                    high_cutoff=0.500)
    
    print('BP done')
    return file_out

def motion_correction(file_in, base, tempor_folder, export_motcor, export_timestamps, export=True):

    # Motion Correction
    print('MC')
    file_out = isx.make_output_file_path(file_in, str(tempor_folder), 'MC')
    isx.motion_correct(file_in, file_out, preserve_input_dimensions=True)
    print('MC done')

    if export:
        print('MC-export')
        isx.export_movie_to_tiff(file_out, str(export_motcor), write_invalid_frames=False)
        isx.export_movie_timestamps_to_csv(file_out, str(export_timestamps), time_ref='start')
        print('MC-export done')
    
    return file_out

def dff(file_in, base, tempor_folder, export_file, export=True):

    # Df/f
    print('DFF')
    file_out = isx.make_output_file_path(file_in, str(tempor_folder), 'DFF')
    isx.dff(file_in, file_out)
    print('DFF done')

    if export:
        print('DFF-export')
        isx.export_movie_to_tiff(file_out, str(export_file), write_invalid_frames=False)
        print('DFF-export done')

    return file_out

def max_projection(file_in, base, tempor_folder, export_file, export=True):

    # max projection
    print('MaxProj')
    file_out = isx.make_output_file_path(file_in, str(tempor_folder), 'maxproj')
    isx.project_movie(file_in, file_out)
    print('MaxProj done')

    if export:
        print('MaxProj-export')
        isx.export_isxd_image_to_tiff(file_out, str(export_file))
        print('MaxProj-export done')

    return file_out

def cnmfe(file_in, base, tempor_folder, export_traces, export_footprints, export=True):

    # CNMFe
    print('CNMFe')
    file_out = isx.make_output_file_path(file_in, str(tempor_folder), 'CNMFe')
    isx.run_cnmfe(file_in, file_out,
        output_dir=str(tempor_folder),
        cell_diameter=7,
        min_corr=0.7,
        min_pnr=7,
        bg_spatial_subsampling=2,
        ring_size_factor=1.4,
        gaussian_kernel_size=0,
        closing_kernel_size=0,
        merge_threshold=0.7,
        processing_mode="parallel_patches",
        num_threads=4,
        patch_size=80,
        patch_overlap=20,
        output_unit_type="df_over_noise")
    print('CNMFe done')


    if export:
        print('Export CNMFe')

        if not file_out.exists():
            print(f"CNMFe produced no cell set file for {base}. Skipping export.")
            return None
        
        cell_set = isx.CellSet.read(str(file_out))
        if cell_set.num_cells == 0:
            print(f"CNMFe found 0 cells for {base}. Skipping export.")
            return None
        
        isx.export_cell_set_to_csv_tiff([str(file_out)], str(export_traces), str(export_footprints))
        print('Export done')

    return file_out


for i, rec in enumerate(rec_files):
        
    base = rec.stem
    rec = str(rec)
    print(f"\n----- {base} -----  file {i}/{len(rec_files)}")

    # paths for exports
    export_preprocess  = export_folder / f"{base}_pp.tif" 
    export_spatband    = export_folder / f"{base}_pp_bp.tif" 
    export_motcor      = export_folder / f"{base}_pp_bp_motcor.tif" 
    export_movietime   = export_folder / f"{base}_pp_bp_motcor_time.csv" 
    export_dff         = export_folder / f"{base}_pp_bp_motcor_dff.tif"
    export_maxproj     = export_folder / f"{base}_pp_bp_motcor_dff_maxproj.tif" 
    export_traces      = export_folder / f"{base}_pp_bp_motcor_CNMFe_traces.csv" 
    export_footprints  = export_folder / f"{base}_pp_bp_motcor_CNMFe_footprints.tif"
     
    # do stuff
    file_pp         = preprocess(rec, base, tempor_folder, export_preprocess)
    file_bp         = spatial_bandpass(file_pp, base, tempor_folder, export_spatband)
    file_motcor     = motion_correction(file_bp, base, tempor_folder, export_motcor, export_movietime)
    file_dff        = dff(file_motcor, base, tempor_folder, export_dff)
    file_max        = max_projection(file_dff, base, tempor_folder, export_maxproj)
    file_cnmfe      = cnmfe(file_motcor, base, tempor_folder, export_traces, export_footprints)
