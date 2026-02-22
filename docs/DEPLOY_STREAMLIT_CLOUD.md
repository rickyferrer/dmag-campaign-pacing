# Deploy to Streamlit Community Cloud

## Prerequisites

1. A GitHub repo containing this project.
2. A Neon Postgres `DATABASE_URL`.

## 1) Push project to GitHub

From your project folder:

```bash
cd "/Users/rickyferrer/Working Files/Codex Project"
git init
git add .
git commit -m "Initial pacing dashboard + pipeline"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## 2) Create Streamlit app

1. Go to https://share.streamlit.io/
2. Sign in with GitHub.
3. Click `New app`.
4. Select:
   - Repository: your repo
   - Branch: `main`
   - Main file path: `dashboard/app.py`
5. Open `Advanced settings`.

## 3) Add secret

In Streamlit `Advanced settings` secrets, paste:

```toml
DATABASE_URL = "postgresql://...your-neon-url..."
```

Then deploy.

## 4) Share access

After deployment, copy the app URL and share with your team.

## Notes

- The dashboard reads from `st.secrets["DATABASE_URL"]` in cloud.
- Local development still uses `.env`.
- Keep ingestion running on a schedule so dashboard data stays current.
