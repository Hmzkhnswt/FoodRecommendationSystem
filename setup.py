import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "FoodRecognition"
AUTHOR_USER_NAME = "HamzaALi"
SRC_REPO = "FoodRecognition"
AUTHOR_EMAIL = "hamzaali.dcse@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A Python library for FOOD Recognition Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},  # Add src as the root directory
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
    ]
)
