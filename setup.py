
import setuptools
#release/distribution instruction from:
#   https://towardsdatascience.com/create-and-publish-your-own-python-package-ea45bee41cdc

# Reads the content of your README.md into a variable to be used in the setup below
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ifitpy',                           # should match the package folder
    packages=['ifitpy'],                     # should match the package folder
    version='0.0.6',                                # important for updates
    license='MIT',                                  # should match your chosen license
    description='Simple data fitting package',
    long_description=long_description,              # loads your README.md
    long_description_content_type="text/markdown",  # README.md is of type 'markdown'
    author='Guilherme Pereira',
    author_email='guipinper@gmail.com',
    url='https://https://github.com/gpinpereira/Pyfit', 
    project_urls = {                                # Optional
        "Bug Tracker": "https://https://github.com/gpinpereira/Pyfit/issues"
    },
    install_requires=['numpy','scipy', 'iminuit'],                  # list all packages that your package uses
    keywords=["fitting", "datascience", "iminuit"], #descriptive meta-data
    classifiers=[                                   # https://pypi.org/classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Documentation',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    
    download_url="https://github.com/gpinpereira/Pyfit/archive/refs/tags/0.0.1.tar.gz",
)