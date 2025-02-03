# INSTALL.md

## Install python dependencies

@MacOS
For compatibility, we temporarily recommend using Python 3.11. To install Python 3.11, use the following command:
```zsh
brew install python@3.11
```
```zsh
echo 'export PATH="/usr/local/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
python3.11 --version
```
This should output the version of Python 3.11 installed on your system.

Create a virtual environment and install the dependencies using the following commands:
```zsh
pip install --upgrade pip
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```