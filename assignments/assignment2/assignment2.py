import subprocess

if __name__ == "__main__":
    datasets =["castle-P19"] # ["castle-P30",  "entry-P10", "fountain-P11", "Herz-Jesus-P8", "Herz-Jesus-P25"] #
    # fetch data from https://github.com/openMVG/SfM_quality_evaluation
    for dataset in datasets:
        # subprocess.run(["python", "assignments/assignment2/feat_match.py", 
        #                 "--data_dir", f'./assets/assignment2/Benchmarking_Camera_Calibration_2008/{dataset}/images', 
        #                 "--out_dir", f'./assets/assignment2/Benchmarking_Camera_Calibration_2008/{dataset}'])
        subprocess.run(["python", "assignments/assignment2/sfm.py", "--dataset", dataset, "--reprojection_thres", "8.0"])