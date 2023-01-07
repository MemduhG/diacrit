# Diacrit

Small repo based on [Shakkelha](https://github.com/AliOsm/shakkelha),
reduced to three files to run as a very basic service.

```commandline
python predict.py -in <input-file> -out <output-file>
```

The three essential files to run this repo are:

```commandline
helpers/
    constants.pkl
models/
    small_rnn.h5
predict.py
```

A very minimal requirements.txt is provided, this should work out of the box as of early 2023.