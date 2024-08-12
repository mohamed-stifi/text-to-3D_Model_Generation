import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"
REPO_NAME = "text-to-3d_model_generation"
AUTHOR_USER_NAME = "Mohamed Stifi"
AUTHOR_EMAIL = "mohamed.stifi.stifi@gmail.com"
PROJECT_NAME = "textTo3DModelGen"

setuptools.setup(
    name=PROJECT_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A sampll python package for creating an app for text to 3d models generation using generative AI.",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),

)