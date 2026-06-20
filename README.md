# The Recruiting Agent Magic (Backend)

Welcome to the brains of the Recruiting Agent! This is where the magic happens. Think of this like a smart robot librarian that reads your company notes, learns how to speak like you, and writes messages to people you want to hire.

## How the Magic Works

The backend follows a very simple story to do its job:

*What makes this a true Agent?* Unlike a simple script that just forwards text to an AI, this system autonomously analyzes the conversation, decides what recruiting stage comes next, retrieves relevant facts on the fly, and rigorously evaluates its own responses against strict policies before ever replying.

*(If you are curious about the technical choices for everything, please refer to the `config.yaml` and `.env` files where all models and configurations are set!)*

1. **Reading Your Notes:** When you type in details about your company (like your culture or what you do), this backend reads it carefully.
2. **Learning the Voice:** It figures out exactly how your company likes to talk. Should it be professional? Friendly? Direct? It learns this just by reading your notes.
3. **Collecting Facts:** It pulls out the most important facts (like "We build AI tools" or "We are looking for builders") and puts them in a special memory box. 
4. **Writing the Messages:** When you say "Write a message to this person," the backend acts like a super-fast recruiter. It reads the person's profile, looks in its memory box for matching facts, and writes a perfect outreach message.
5. **Having a Conversation:** When you pretend to be the candidate and reply, the backend reads your reply, decides the best thing to say next, and writes back instantly!

*Note: The backend never invents things! It only ever uses the facts it pulled from your original notes.*

## How to Get Started

Getting the robot running on your computer is super easy! Just follow these steps:

### 1. Set Up the Files
You need to create two special files so the backend knows the rules. Don't worry, we gave you templates!

1. Copy the file called `config.yaml.example` and name the new copy `config.yaml`.
2. Copy the file called `.env.example` and name the new copy `.env`.
3. Open the `.env` file you just made and put in the secret keys (your API keys). 

### 2. Start the Engine
Open your terminal (the black screen where you type commands) and type this exactly:

```bash
uv sync --all-groups
uv run uvicorn psview_agent.main:app --host 0.0.0.0 --port 8000
```

*Boom!* The backend is now awake and listening on your computer at `http://localhost:8000`. 

## How the Frontend Talks to the Backend

The frontend website you see on your screen talks to this backend using special "API" calls. 
- It asks `/ready` to see if the robot is awake.
- It sends your notes to `/configure` to let the robot learn about your company.
- It asks `/start` to generate the first messages.
- It sends your chat replies to `/turn` to get a response back.

That's it! Everything works together seamlessly to give you a magical recruiting experience!
