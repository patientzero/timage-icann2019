import click
from pathlib import Path
import logging
from .data.image import image_classes
from .network import network_models_classes

CLI_NAME = "timage"


def setup_logging(log_level):
    ext_logger = logging.getLogger("py.warnings")
    logging.captureWarnings(True)
    level = getattr(logging, log_level)
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(filename)s: %(message)s", level=level)
    if level <= logging.DEBUG:
        ext_logger.setLevel(logging.WARNING)


@click.group()
@click.option("-l", "--log-level", default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']))
def cli(log_level):
    setup_logging(log_level)


@cli.command()
@click.option("--image-dir", help="UCR data directory", required=True)
@click.option("--log-dir", help="Output directory", required=True)
@click.option("--weights-path", help="Weights file")
@click.option("--resolution", "-r", help="Image resolution , e.g. 256x256", required=True)
@click.option("--depth", "-d", help="Image color depth", default=1)
@click.option("--image-cls", help='recu or other implementations',
                type=click.Choice(image_classes.keys()), required=True)
@click.option("--network-model", type=click.Choice(network_models_classes.keys()))
@click.option("--nb-epoch", help="Number of epochs to train, e.g. 50", default=50)
@click.option("--batch-size", help="Training batch size, e.g. 8", default=8)
@click.option("--datasets", help="Dataset comma separated, e.g. Cricket_X,50Words")
@click.option("--use-class-weight", help="Train using class weight", is_flag=True)
def train(image_dir, log_dir, weights_path, resolution, depth, image_cls, network_model, nb_epoch,
          batch_size, datasets, use_class_weight):
    from .data.image.loader import UCRImageLoader
    from datetime import datetime
    from pathlib import Path

    log_dir = Path(log_dir) / datetime.now().strftime("%y-%m-%d_%H-%M")
    img_rows, img_cols = list(map(int, resolution.split("x")))
    image_cls = image_classes[image_cls]
    network_model_cls = network_models_classes[network_model]

    loader = UCRImageLoader.from_path(image_dir, (img_rows, img_cols), depth, image_cls)
    datasets = loader.names if datasets is None else datasets.split(',')

    for name in datasets:
        try:
            data = loader.load(name)
            num_classes = data.train_labels.shape[1]
            network = network_model_cls(image_cls, img_rows, img_cols, depth, num_classes, weights_path)
            network.fit(data.train_data, data.train_labels,
                        validation_data=(data.test_data, data.test_labels),
                        name=name,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        base_log_dir=log_dir,
                        class_weight=data.train_class_weight if use_class_weight else None)
        except FileNotFoundError:
            logging.error("Images not found: {}".format(name))
        except Exception as e:
            logging.exception(e)
        finally:
            del network


@cli.command()
@click.option("--data-dir", help="UCR data directory", required=True)
@click.option("--image-dir", help="Output directory", required=True)
@click.option("--resolution", "-r", help="Image resolution , e.g. 256x256", required=True)
@click.option("--depth", "-d", help="Image color depth", default=1)
@click.option("--image-cls", help='recu or other implementations',
                type=click.Choice(image_classes.keys()), required=True)
@click.option("--datasets", help="Datset comma separated, e.g. Cricket_X,50Words")
@click.option("--njobs", 'n_jobs', help="Number of parallel jobs, defaults to number of CPUs", default=None)
@click.option("--z-norm", help="If enabled the data is z-normalized", is_flag=True)
def create_images(data_dir, image_dir, resolution, depth, image_cls, datasets, n_jobs, z_norm):
    from ucrloader import UCRLoader
    from joblib import Parallel, delayed
    from multiprocessing import cpu_count
    from .data.image.loader import UCRImageLoader

    image_cls = image_classes[image_cls]

    img_rows, img_cols = list(map(int, resolution.split("x")))
    ucr_loader = UCRLoader(data_dir)
    datasets = ucr_loader.names if datasets is None else datasets.split(',')

    # how many jobs?
    n_jobs = cpu_count() if n_jobs is None else int(n_jobs)
    n_jobs = n_jobs if len(datasets) > n_jobs else len(datasets)
    logging.info("Using {} jobs for {} datasets".format(n_jobs, len(datasets)))

    Parallel(n_jobs=n_jobs)(delayed(UCRImageLoader.convert)(
        ucr_loader,
        image_dir,
        (img_rows, img_cols),
        depth,
        image_cls,
        [name],
        z_norm) for name in datasets)


@cli.command()
@click.option("--image-dir", help="Output directory", required=True)
@click.option("--resolution", "-r", help="Image resolution , e.g. 256x256", required=True)
@click.option("--depth", "-d", help="Image color depth", default=1)
@click.option("--image-cls", help='recu or other implementations',
                type=click.Choice(image_classes.keys()), required=True)
@click.option("--datasets", help="Datset comma separated, e.g. Cricket_X,50Words")
def check_images(image_dir, resolution, depth, image_cls, datasets):
    from .data.image.loader import UCRImageLoader, UCRImage

    image_cls = image_classes[image_cls]

    img_rows, img_cols = list(map(int, resolution.split("x")))

    base_path = Path(UCRImageLoader.dir_name(image_dir, (img_rows, img_cols), depth, image_cls))
    datasets = [f.name for f in base_path.glob('*')] if datasets is None else datasets.split(',')

    def check(path):
        labels = path / UCRImage.LABELS_FILE
        try:
            labels_count = len(labels.open().readlines())
        except FileNotFoundError:
            logging.error("Error: labels missing", path.absolute())
            return
        images_count = len(list(path.glob('*' + UCRImage.FILE_EXTENSION)))

        if labels_count != images_count:
            logging.error("Error: images missing", path.absolute(), labels_count, images_count)

    for d in datasets:
        path = base_path / d
        check(path / 'train')
        check(path / 'test')


@cli.command()
@click.option("--log-dir", help="Output directory", required=True)
def results(log_dir):
    from .results import ResultParser
    csv = ResultParser(log_dir).as_csv()
    print(csv)


@cli.command()
@click.option("--model-path", help="dir with complete model trained on all datasets")
@click.option("--weights-path", help="dir with complete model trained on all datasets")
@click.option("--network-model", type=click.Choice(network_models_classes.keys()))
@click.option("--label-dict", help="Absolute path to a label dictionary csv")
@click.option("--image-dir", help="UCR data directory", required=True)
@click.option("--image-cls", help='recu or other implementations',
                type=click.Choice(image_classes.keys()), required=True)
@click.option("--resolution", "-r", help="Image resolution , e.g. 256x256", required=True)
@click.option("--depth", "-d", help="Color depth of image, e.g. 3", default=1)
@click.option("--out-file", help="Name of output file, if None print to standard output")
@click.option("--datasets", help="Dataset comma separated, e.g. 50Words,Adiac")
def single_results(model_path, weights_path, network_model, label_dict,
                   image_dir, image_cls, resolution, depth, out_file, datasets):
    from keras.engine.saving import load_model
    from .data.image.loader import UCRImageLoader
    from .data.all import load_labeldict
    from .results import SingleResultsParser
    from .network.base import LoadedNetworkModel

    img_rows, img_cols = list(map(int, resolution.split("x")))
    image_cls = image_classes[image_cls]

    label_path = Path(label_dict)

    label_dict = load_labeldict(label_path)
    loader = UCRImageLoader.from_path(image_dir, (img_rows, img_cols), depth, image_cls)
    if datasets is not None:
        datasets = datasets.split(',')

    if model_path and not weights_path:
        network = LoadedNetworkModel()
        network.model = load_model(str(Path(model_path)))
    elif not model_path and weights_path and network_model:
        network_model_cls = network_models_classes[network_model]
        num_classes = sum(len(d.keys()) for _, d in label_dict.items())
        network = network_model_cls(image_cls, img_rows, img_cols, depth, num_classes, weights_path)
        network.load_weights(by_name=True)

    res = SingleResultsParser(loader, network, label_dict)
    res.as_csv(datasets, out_file)


@cli.command()
@click.option("--input-data-dir", help="absolute path to image dir", required=True)
@click.option("--output-data-dir", help="storage location of all datadir", required=True)
def make_all_data_dir(input_data_dir, output_data_dir):
    from .data.all import make_all_data_dir
    make_all_data_dir(input_data_dir, output_data_dir)


@cli.command()
@click.option("--experiment-dir", help="dir with complete model trained on all datasets", required=True)
@click.option("--image-dir", help="UCR data directory", required=True)
@click.option("--network-model", type=click.Choice(network_models_classes.keys()), required=True)
@click.option("--image-cls", help='recu or other implementations',
                type=click.Choice(image_classes.keys()), required=True)
@click.option("--resolution", "-r", help="Image resolution , e.g. 256x256", required=True)
@click.option("--depth", "-d", help="Color depth of image, e.g. 3", default=1)
@click.option("--out-dir", help="Name of output file, if None print to standard output", required=True)
@click.option("--data-sets", help="Dataset comma separated, e.g. 50Words,Adiac", default=None)
@click.option("--label-dict", help="Absolute path to a label dictionary csv", default=None)
def conf_matrices(experiment_dir, image_dir, network_model, image_cls, resolution, depth, out_dir,
                  data_sets, label_dict):
    from .results import GraphicalResults
    from timage.data.all import load_labeldict

    if label_dict is not None:
        label_dict = load_labeldict(Path(label_dict))
    result = GraphicalResults(experiment_dir, image_dir, network_model,
                              image_cls, resolution, depth,
                              lbl_dict=label_dict)
    if data_sets is not None:
        data_sets = data_sets.split(',')
    result.confusion_matrix(out_dir, data_sets)
