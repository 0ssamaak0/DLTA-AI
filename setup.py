import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
    
__version__ = "0.0.0"
with open("DLTA_AI_app/labelme/__init__.py", "r") as f:
    for line in f.readlines():
        if line.startswith("__version__"):
            __version__ = line.split("=")[1].strip().strip('"')
            break

setuptools.setup(
    name="DLTA-AI",
    version=f"{__version__}",
    author="0ssamaak0",
    author_email="0ssamaak0@gmail.com",
    description="DLTA-AI is the next generation of annotation tools, integrating the power of Computer Vision SOTA models to Labelme in a seamless expirence and intuitive workflow to make creating image datasets easier than ever before",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/0ssamaak0/DLTA-AI",
    package_dir={"DLTA_AI_app": "DLTA_AI_app"},
    python_requires='>=3.8',
    install_requires=requirements,
    package_data={"": ["*"]},
    license="GPLv3",
    entry_points={
        "console_scripts": [
            "DLTA-AI=DLTA_AI_app.__main__:main"
        ]
    }
)

