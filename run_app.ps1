# Run the Streamlit app (ASCII-safe entrypoint)
# Disable first-run email prompt to avoid interactive failure in conda run.
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
python -m streamlit run app\app_ascii.py @args
