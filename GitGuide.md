# Git Guide

# Clone the repository
```
git clone https://github.com/FlammingFrost/puma.git
cd puma
```

# Branch workflow
Never work directly on the main branch, use feature branches.
1. **Update your local main branch first**
```
git checkout main
git pull origin main
```
2. Create a new feature branch
```
git checkout -b feature/my-feature
```
or enter an existing branch
```
git checkout feature/my-feature
```
3. Make changes
\* *Some work* \*
Commit your changes
```
git add .
git commit -m "commit message"
```
Push your changes
```
git push origin feature/my-feature
```
Create a pull request on GitHub and wait for approval.
	1.	On GitHub, go to the repository.
	2.	Click Pull requests. (Code > Pull requests)
	3.	Add a description of the changes.
	4.	Click Create Pull Request.
Once approved, **you** merge the pull request on **GitHub**.




