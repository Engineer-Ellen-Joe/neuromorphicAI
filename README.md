<div align="center">
  
  # Ellen Project for Public
  
A hybrid AI under development, inspired by neurons and synapses in the brain using SNN-based architecture.
</div>

## Dev Notes
*This is still an early version!*
> **[main (stable)](https://github.com/Ellen-project/Public-repository/tree/main)**  
> + Only stable code without syntax errors will be pushed.

> **[dev (preview)](https://github.com/Ellen-project/Public-repository/tree/dev)**  
> + Contains upcoming features and bug fixes.  
> + May run slower due to optimization issues.  
> + Code with syntax errors will not be pushed.

> **[testDev (experimental)](https://github.com/Ellen-project/Public-repository/tree/testDev)**
> + Unverified code will be uploaded here.
> + Experimental structural changes to neuron models and CUDA kernels will be uploaded.
> + This is a personal research playground.

## Usage

### 1. Required Dependencies
- Python 3.12.10 (or lower if needed)
- CUDA Toolkit 12.9 (or lower if needed)
- Java Development Kit

### 2. Install Python Packages
Install the required Python packages using:

```
pip3 install -r requirements.txt
```

### ⚠️ If you get an error log
```
ERROR: Could not find a version that satisfies the requirement torch==2.8.0+cu129 (from versions: 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0, 2.7.0, 2.7.1, 2.8.0)
```
visit [PyTorch](https://pytorch.org/) website and install the Stable (2.8.0) version with CUDA 12.9.

## CLI CMD
|Command|Description|
|:---|:---|
|```CTRL + C```|Shutdown the model|

## Open Source Acknowledgment

Ellen Project uses components from the following open-source project:

- **Open Korean Text (Okt)** — Licensed under the **Apache License 2.0**  
  Source: [https://github.com/open-korean-text/open-korean-text](https://github.com/open-korean-text/open-korean-text)

The Okt library is included in this project as a pre-built `.jar` file under  
`src/external/okt/`, along with its original LICENSE and README files.

see [NOTICE](NOTICE) for details.
