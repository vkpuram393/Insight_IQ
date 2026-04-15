"""
Minimal setup.py for CI/CD build compatibility.
This project is an application, not a distributable package.
"""

from setuptools import setup, find_packages

setup(
    name="pss-myclaims-ai-agent",
    version="0.1.0",
    description="PBM AI Assist – LangGraph Multi-Agent Starter",
    author="CVS Health",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=[
        # Dependencies are managed via requirements.txt
        # This is just for CI/CD compatibility
    ],
    include_package_data=True,
    zip_safe=False,
)
