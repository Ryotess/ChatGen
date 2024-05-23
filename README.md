<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Ryotess/ChatGen">
    <img src="images/logo.png" alt="Logo" width="512" height="256">
  </a>

<h3 align="center">ChatGen--An Algorithm to generate coversation history for training llm/chatbot</h3>

  <p align="center">
    This is an algorithm that can help you generate your own custom conversation dataset with history to be used to train LLM/ChatBot
    <br />
    <a href="https://github.com/Ryotess/ChatGen"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Ryotess/ChatGen/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/Ryotess/ChatGen/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#quick-start">Quick Start</a></li>
      </ul>
    </li>
    <li><a href="#instruction">Instruction</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
[![Python][Python]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Please follow the instructions to install and set up the environment for this project.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Ryotess/ChatGen.git
   ```
2. Move to the project
   ```sh
   cd ./ChatGen
   ```
3. Build environment
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

###  Quick Start
Here is the quick start demo base on sample dataset, if you wanna use this project on your own dataset, please read the [instruction](https://github.com/Ryotess/ChatGen/blob/main/instruction.ipynb)

```sh
# Import packages
from chatgen.chat_algo import ChatAlgo
from chatgen.data_loader import load_xlsx, create_input_data
```
```sh
# Set data path
input_file = "./dataset/sample_dataset.xlsx" # sample dataset
sheet_name = 'QuestionAskingMerge' # the sheet we would use
output_file = "./output/conversations.json" # output path(please remember to create an ./output directory)
```

```sh
# Load raw data & Create input data
data = load_xlsx(input_file, sheet_name)
input_data = create_input_data(data)
```
```sh
# Create conversation dataset
chat_algo = ChatAlgo(input_data) # initialization
chat_algo.create_chat_history() # generate
```

```sh
# save to JSON
chat_algo.to_json(output_file)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Instruction
Here we provide a brief instruction of our algorithm design in this project and sample demo.  
Please open the [instruction](https://github.com/Ryotess/ChatGen/blob/main/instruction.ipynb) and follow the steps to get a more comprehensive understand of this project.



<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

E-Mail: jessforwork2023@gmail.com

Project Link: [https://github.com/Ryotess/ChatGen](https://github.com/Ryotess/ChatGen)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/Ryotess/ChatGen.svg?style=for-the-badge
[license-url]: https://github.com/Ryotess/ChatGen/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/shaoyanchen
[product-screenshot]: images/screenshot.png
[Python]: https://img.shields.io/pypi/pyversions/numpy
[Python-url]: https://numpy.org/
