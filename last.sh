# python main.py --db MOTHERS --selcids 4 --fxs 996 --cfg nogravity_hack:%1 --vecfield --nocache-feat --nogui
# python main.py --db MOTHERS --selcids 4 --fxs 946 --cfg nogravity_hack:%1 --vecfield --nocache-feat --nogui
# python main.py --db MOTHERS --selcids 4 --fxs 882 --cfg nogravity_hack:%1 --vecfield --nocache-feat --nogui


python dev.py --db GZ --nocache-feat --cfg nogravity_hack:True --all-cases --preload
python dev.py --db MOTHERS --nocache-feat --cfg nogravity_hack:True --all-cases --preload

python dev.py --db GZ -t gravity_test --all-gt-cases
python dev.py --db MOTHERS -t gravity_test --all-gt-cases


export DB=GZ
export DB=MOTHERS
python dev.py --allgt --db $DB -t gravity_test
python dev.py --allgt --db $DB --stdcfg gravity_test 0 -t scores
python dev.py --allgt --db $DB --stdcfg gravity_test 1 -t scores

python main.py --allgt --db $DB --stdcfg gravity_test 0 -t scores
