# Nekrasovka.ru newspaper recognition system
recognition_scalable

---
## Docker

- Pull repo:
```
git clone <repo>
cd <repo folder>
```
- Build docker image:
```
sh prepare.sh
```

- Utilize containers using the following command in Python (Python utilizes only default libraries here):
```
python run.py -i test_data/ToRecognize -o test_data/Recognized -j 4
```
