
pip install -r requirements.txt
set iterations=3

for /l %%i in (1,1,%iterations%) do (
    echo Iteration: %%i
    python main_kd.py
)