from collections import OrderedDict
from pathlib import Path
import shutil
from .image.loader import UCRImage


def load_labeldict(path):
    labels = [l.split(';') for l in path.read_text().split('\n')][:-1]
    label_dict = {}
    for l in labels:
        if l[0] in label_dict.keys():
            label_dict[l[0]][int(l[1])] = int(l[2])
        else:
            label_dict[l[0]] = {}
            label_dict[l[0]][int(l[1])] = int(l[2])
    return label_dict


def lbldict_as_csv(labeldict, path):
    csv = ""
    for dataset in labeldict.keys():
        for k, v in labeldict[dataset].items():
            csv += "{};{};{}\n".format(dataset, k, v)
    with (path / 'labeldict.csv').open('w') as file:
        file.write(csv)


def make_all_data_dir(input_data_dir, output_data_dir):
    p = Path(input_data_dir)
    # make dependent of path, target a UCR Image dir, create path completely
    tp = (Path(output_data_dir) / p.name / 'all')
    tp.mkdir(exist_ok=True, parents=True)
    datasets = sorted([x.name for x in p.iterdir() if x.is_dir()])
    labeldict = OrderedDict()
    # get unique labels and make new labels
    for part in [UCRImage.TRAIN_DIR, UCRImage.TEST_DIR]:
        (tp / part).mkdir(exist_ok=True)
        for data in datasets:
            with (p / data / part / UCRImage.LABELS_FILE).open('r') as file:
                labels = file.readlines()
            flavours = list(set(int(x) for x in labels))
            flavours.sort()
            labeldict[data] = flavours
        i = 0
        for data in datasets:
            lbls = labeldict[data]
            labeldict[data] = {}
            for lbl in lbls:
                labeldict[data][lbl] = i
                i = i + 1
        lbldict_as_csv(labeldict, tp / part)

    # translate labels
    for part in [UCRImage.TRAIN_DIR, UCRImage.TEST_DIR]:
        all_labels = []
        for data in datasets:
            with (p / data / part / UCRImage.LABELS_FILE).open('r') as file:
                labels = file.readlines()
            values = [int(x) for x in labels]
            all_labels += [labeldict[data][v] for v in values]
        (tp / part / UCRImage.LABELS_FILE).write_text('\n'.join(map(str, all_labels)))

    for part in [UCRImage.TRAIN_DIR, UCRImage.TEST_DIR]:
        i = 0
        for data in datasets:
            sourcepath = p / data / part
            targetpath = tp / part
            files = [x.name for x in sorted(sourcepath.iterdir()) if UCRImage.FILE_EXTENSION == x.suffix]
            for f in files:
                source = sourcepath / f
                dest = targetpath / (str(i).zfill(8) + UCRImage.FILE_EXTENSION)
                shutil.copy(str(source), str(dest))
                i += 1
