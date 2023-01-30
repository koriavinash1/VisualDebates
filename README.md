# Visual Debates
[![arXiv](https://img.shields.io/badge/arXiv-2008.06457-<COLOR>.svg)](https://arxiv.org/abs/2210.09015)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Installation

```
git clone https://github.com/koriavinash1/VisualDebates.git
cd VisualDebates
pip install -r requirements.txt
```

<hr>

## Citation

If you use this work in your research, please consider citing our work:

```
@article{kori2022visual,
  title={Visual Debates},
  author={Kori, Avinash and Glocker, Ben and Toni, Francesca},
  journal={arXiv preprint arXiv:2210.09015},
  year={2022}
}
``` 


## Data folder structure

```
Dataset
	- training
		- <class1>
			- <img1.png>
			- <img2.png>
		- <class2>
			- <img1.png>
			- <img2.png>
		- context-data.csv
	- validation 
		- <class1>
		- <class2>
		- context-data.csv
	- testing 
		- <class1>
		- <class2>
		- context-data.csv
		
``` 



## Reproducibility

```
cd scripts
. exps.sh  # NOTE: This command will execute 24 different ablation exp. it is recommened to select exp. before running
```

  
## Contributing
Contributions are what make the open source community such an amazing place to be learn, inspire, and create. 
Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request
  
## Contact 
* Avinash Kori (avin.kori.re@gmail.com)

