@echo off
REM Redirect all output to output.txt
echo Run the project > output.txt
python main.py >> output.txt 2>&1
echo Script completed successfully. >> output.txt