import os
from argparse import ArgumentParser
import labelbox.exporters.coco_exporter as lb2coco
import labelbox.exporters.voc_exporter as lb2pa


def build_parser():
    """Creates parser
    Returns
    -------
    args : dictionary
        A dictionary containing command line arguments to test processes.

    Examples
    --------
    To convert json labels run:
    ``python jsonlabelbox.py input_file.json output_directory -f VOC``
    """
    parser = ArgumentParser(description="Converts LabelBox json files to COCO or VOC format.")

    parser.add_argument("input_file", help="Input json file")
    parser.add_argument("output_dir", nargs="*", default=os.getcwd(), help="Output directory")
    parser.add_argument("-f", "--format", dest="format", default="VOC", choices=["COCO", "VOC"],
                        help="Output format")
    return parser


def parse_arguments(parser):
    return parser.parse_args()


def json2coco(input_file, output_dir):
    labeled_data = input_file  # file path to JSON export with XY paths
    coco_output = os.path.join(output_dir, "-".join(["coco", input_file]))  # where to write COCO output
    lb2coco.from_json(labeled_data, coco_output, label_format='XY')


def json2voc(input_file, output_dir):
    labeled_data = input_file  # file path to JSON export with XY paths
    # where to write VOC annotation and image outputs
    annotations_output_dir = output_dir
    images_output_dir = output_dir
    lb2pa.from_json(labeled_data, annotations_output_dir, images_output_dir, label_format='XY')


def main():
    parser = build_parser()
    arguments = parse_arguments(parser)

    if arguments.format.upper() == "COCO":
        print("Converting to COCO format...")
        json2coco(arguments.input_file, arguments.output_dir)

    if arguments.format.upper() == "VOC":
        print("Converting to VOC format...")
        json2voc(arguments.input_file, arguments.output_dir)

    print("...done!")


if __name__ == '__main__':
    main()
