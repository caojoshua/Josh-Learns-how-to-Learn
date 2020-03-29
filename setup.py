import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JoshLearnsHowToLearn",
    version="0.0.1",
    author="Joshua Cao",
    author_email="cao.joshua@yahoo.com",
    description="A simple deep learning framework",
	long_description=long_description,
    url="https://github.com/caojoshua/Josh-Learns-how-to-Learn",
    packages=['JoshLearnsHowToLearn'],
    install_requires=[
        "numpy",
		"sklearn"
    ],
	test_suite='nose.collector',
    tests_require=['nose'],
)