import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
from pydub import AudioSegment
import noisereduce as nr  
from tqdm import tqdm  
import argparse
import shutil


# Set these depending on dataset and functions desired
INPUT_DIR = '/home/ubuntu/arizonafiles/voice_clone/VoiceDatasetCreation/outputP/wavs'
#BASE_OUTPUT_DIR = '/Users/miria/Downloads/output'
input_dir_name = os.path.basename(os.path.normpath(INPUT_DIR))





# Processing settings
PEAK_DBFS_MIN = -10  # Lower bound for peak dBFS
PEAK_DBFS_MAX = -7   # Upper bound for peak dBFS
TARGET_SAMPLE_RATE = 22050  # Set target sample rate (22050 is VITS compatible)

# Define output directories for each mode
# OUTPUT_DIRS = {
#     "normalize": os.path.join(BASE_OUTPUT_DIR, "normalized_wavs"),
#     "noise_reduction": os.path.join(BASE_OUTPUT_DIR, "noise_reduced_wavs"),
#     "resampling": os.path.join(BASE_OUTPUT_DIR, "resampled_wavs"),
# }
def copy_unchanged_audio(input_path, output_path):
    """Copy an audio file to the output directory if no processing is needed."""
    if os.path.abspath(input_path) != os.path.abspath(output_path):  # Avoid self-copying
        shutil.copy(input_path, output_path)
        print(f"Copied {os.path.basename(input_path)} to {output_path} (No processing needed)")


def resample_audio(input_path, output_dir, target_sample_rate=TARGET_SAMPLE_RATE):
    """Resample a WAV file to the target sample rate and save it in output_dir."""
    audio, sample_rate = sf.read(input_path)
    output_file = os.path.join(output_dir, os.path.basename(input_path))  # Ensure correct file path

    if sample_rate == target_sample_rate:
        copy_unchanged_audio(input_path, output_file)  # Copy unchanged file
        return

    resampled_audio = signal.resample_poly(audio, target_sample_rate, sample_rate)
    sf.write(output_file, resampled_audio, target_sample_rate)  # Write to the correct file
    print(f'Resampled {os.path.basename(input_path)} to {target_sample_rate} Hz')



def noise_reduction(audio, sample_rate):
    """Apply noise reduction with added dithering to avoid zero-division issues."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono if needed

    if np.max(np.abs(audio)) < 1e-4:  # If audio is too quiet, skip noise reduction
        print('Skipping noise reduction due to low signal level.')
        return audio

    # Avoid divide-by-zero 
    audio += np.random.normal(0, 1e-6, audio.shape)

    return nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.85)


def process_audio_file(file_path, output_dir):
    """Load, apply noise reduction, and save the processed audio."""
    try:
        audio, sample_rate = sf.read(file_path)
        cleaned_audio = noise_reduction(audio, sample_rate)

        output_file = os.path.join(output_dir, os.path.basename(file_path))
        sf.write(output_file, cleaned_audio, sample_rate)

        print(f'Noise-reduced audio saved at: {output_file}')
    except Exception as e:
        print(f'Error processing {file_path}: {e}')


def measure_peak_dbfs(audio_file):
    """Measure the peak dBFS of an audio file correctly handling integer and float formats."""
    try:
        audio, sample_rate = sf.read(audio_file, always_2d=True)
        audio = audio[:, 0]  # Use only one channel if stereo
        peak_amplitude = np.max(np.abs(audio))  # Find peak amplitude

        # Check if the audio is integer-based (e.g., int16, int32)
        if np.issubdtype(audio.dtype, np.integer):
            max_possible_amplitude = np.iinfo(audio.dtype).max  # e.g., 32767 for int16
        else:
            max_possible_amplitude = 1.0  # Float WAVs are usually in range [-1,1]

        # Compute peak dBFS relative to max amplitude
        peak_dbfs = 20 * np.log10(peak_amplitude / max_possible_amplitude) if peak_amplitude > 0 else -np.inf
        return peak_dbfs
    except Exception as e:
        print(f"Error measuring peak dBFS for {audio_file}: {e}")
        return None


def normalize_audio(file_path, output_dir):
    """Normalize audio to be within -7 dBFS to -10 dBFS and print dBFS before and after normalization."""
    print(f"Processing: {file_path}")
    current_peak_dbfs = measure_peak_dbfs(file_path)

    if current_peak_dbfs is None:
        print(f"Skipping {file_path}: Peak dBFS measurement failed.")
        copy_unchanged_audio(file_path, os.path.join(output_dir, os.path.basename(file_path)))
        return

    print(f"Original Peak dBFS: {current_peak_dbfs:.2f} dB")  # Print original dBFS

    if PEAK_DBFS_MIN <= current_peak_dbfs <= PEAK_DBFS_MAX:
        print(f"{file_path} is already within the range (-7 to -10 dBFS). Skipping.")
        copy_unchanged_audio(file_path, os.path.join(output_dir, os.path.basename(file_path)))
        return

    # Determine the target peak dBFS
    target_peak_dbfs = PEAK_DBFS_MAX if current_peak_dbfs < PEAK_DBFS_MIN else PEAK_DBFS_MIN
    gain = target_peak_dbfs - current_peak_dbfs  # Gain in dB to apply

    # Load audio and apply gain
    audio = AudioSegment.from_file(file_path)
    normalized_audio = audio.apply_gain(gain)

    output_file = os.path.join(output_dir, os.path.basename(file_path))
    normalized_audio.export(output_file, format="wav")

    # Measure dBFS after normalization
    new_peak_dbfs = measure_peak_dbfs(output_file)
    print(f"Saved normalized: {output_file} (Applied Gain: {gain:.2f} dB, New Peak dBFS: {new_peak_dbfs:.2f} dB)")




from scipy.io import wavfile
import glob

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--mode', type=str, required=True, choices=['normalize', 'noise_reduction', 'resampling', 'all'])
    args = args_parser.parse_args()

    # Set the correct output directory based on mode
    OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_DIR), f"{input_dir_name}_{args.mode}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_list = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav")]

    if args.mode == 'noise_reduction':
        for file_name in tqdm(file_list, desc="Noise Reduction", unit="file"):
            process_audio_file(os.path.join(INPUT_DIR, file_name), OUTPUT_DIR)
        print('Noise reduction complete!')

        # Dynamically select the first WAV file in the output directory
        wav_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")))
        if wav_files:
            sample_rate, _ = wavfile.read(wav_files[0])  # Read the first WAV file
            print(f"Sample rate: {sample_rate} Hz")
        else:
            print("No WAV files found in the output directory.")

    elif args.mode == 'resampling':
        for file_name in tqdm(file_list, desc="Resampling", unit="file"):
            input_file = os.path.join(INPUT_DIR, file_name)
            resample_audio(input_file, OUTPUT_DIR)  # Pass directory instead of file path
        print('Resampling complete!')
        
        # Dynamically select the first WAV file in the output directory
        wav_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")))
        if wav_files:
            sample_rate, _ = wavfile.read(wav_files[0])  # Read the first WAV file
            print(f"Sample rate: {sample_rate} Hz")
        else:
            print("No WAV files found in the output directory.")

    elif args.mode == 'normalize':
        for file_name in tqdm(file_list, desc="Normalization", unit="file"):
            normalize_audio(os.path.join(INPUT_DIR, file_name), OUTPUT_DIR)
        print("Normalization complete!")

        # Dynamically select the first WAV file in the output directory
        wav_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")))
        if wav_files:
            sample_rate, _ = wavfile.read(wav_files[0])  # Read the first WAV file
            print(f"Sample rate: {sample_rate} Hz")
        else:
            print("No WAV files found in the output directory.")


    elif args.mode == 'all':
        all_dir = os.path.join(os.path.dirname(INPUT_DIR), f"{input_dir_name}_all")
        os.makedirs(all_dir, exist_ok=True)
        temp_dir = all_dir

        for file_name in tqdm(file_list, desc="Normalization", unit="file"):
            normalize_audio(os.path.join(INPUT_DIR, file_name), temp_dir)
        for file_name in tqdm(file_list, desc="Noise Reduction", unit="file"):
            process_audio_file(os.path.join(temp_dir, file_name), temp_dir)
        for file_name in tqdm(file_list, desc="Resampling", unit="file"):
            resample_audio(os.path.join(temp_dir, file_name), temp_dir)
        print("All processing complete!")

    # Ensure every file from the input directory is present in the output directory
    input_files = set(file_list)
    output_files = set(f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".wav"))

    # missing_files = input_files - output_files
    # if missing_files:
    #     print("Warning: Some files were not processed, copying them over unchanged.")
    #     for missing_file in missing_files:
    #         copy_unchanged_audio(os.path.join(INPUT_DIR, missing_file), os.path.join(OUTPUT_DIR, missing_file))

    print(f"Processing complete! Total input files: {len(input_files)}, Total output files: {len(output_files)}")