## Terminal Setup & Installation Guide

Since this application runs on a local environment, please follow the terminal setup instructions below to configure the environment and run the application.

### 1. Clone the Repository & Navigate to the Directory

Open your terminal and clone this repository, then move into the project folder:

```bash

git clone [[https://github.com/b11nnn/Animation-Highlight-Detection-Dynamics-Using-Character-Emotion.git](https://github.com/b11nnn/Animation-Highlight-Detection-Dynamics-Using-Character-Emotion.git)](https://github.com/b11nnn/Animation-Highlight-Detection-Dynamics-Using-Character-Emotion.git](https://github.com/b11nnn/Animation-Highlight-Detection-Dynamics-Using-Character-Emotion.git))

cd Animation-Highlight-Detection-Dynamics-Using-Character-Emotion
```

### 2. Create and Activate a Virtual Environment (Recommended)

To prevent dependency conflicts, it is highly recommended to set up a virtual environment:

- **Mac / Linux:**

```bash
python3 -m venv venv
  source venv/bin/activate
```

- **Windows (Command Prompt):**

```bash
python -m venv venv
  call venv\Scripts\activate
```

### 3. Install Required Dependencies

Install all the necessary packages and libraries specified in the requirements file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

Once the installation is complete, launch the local Streamlit server with the following command:

```bash
streamlit run [app.py](http://app.py)
```

After running this command, your default web browser will automatically open the application at `http://localhost:8501`.