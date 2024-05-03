
pip install -r requirements.txt

set "networks=ESPCN"
set "quantised=true"
set "bit_widths=8"
set iterations=3

for /l %%i in (1,1,%iterations%) do (
    echo Iteration %%i
    for %%n in (%networks%) do (
        for %%b in (%quantised%) do (
            if %%b == true (
                for %%w in (%bit_widths%) do (
                    python create_networks.py %%n %%b %%w
                )
            ) else (
                python create_networks.py %%n %%b
            )
        )
    )
)
