# Environments setup

## 1. Install Anaconda
- Follow the basic install instruction [Link](https://docs.anaconda.com/anaconda/install/)
- (For Windows)
    - `$ curl https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe --output .\Downloads\Anaconda3-2024.10-1-Windows-x86_64.exe`
- (For Mac, Apple Silicon)
    - `$ curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-MacOSX-arm64.sh`
    - `$ bash Anaconda3-2024.10-1-MacOSX-arm64.sh`

## 2. Install dependencies
```bash
$ conda env create --file environment.yaml
$ conda activate aue8089pa1
```

## 3. Install VSCode (Optional)
- Download: https://code.visualstudio.com/download
