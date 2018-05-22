import argparse
import os
import shutil
import tempfile

import torch


def restricted_float(min, max):
    def restricted(x):
        x = float(x)
        if x < min or x > max:
            raise argparse.ArgumentTypeError("{} not in range [{}, {}]".format(x, min, max))
        return x

    return restricted


def restricted_str(*vals):
    def restricted(x):
        x = str(x)
        if x not in vals:
            raise argparse.ArgumentTypeError("{} not in {}}".format(x, vals))
        return x

    return restricted


def restricted_list(*vals, sep='-'):
    def restricted(param):
        params = param.split(sep)
        # print(params)
        # for x in params:
        #     if x not in vals:
        #         raise argparse.ArgumentTypeError("{} not in {}}".format(x, vals))
        return params

    return restricted


def split_list(param, elt_type=str, sep='-'):
    assert isinstance(param, str)
    return [elt_type(p) for p in param.split(sep)]


def save_checkpoint(checkpoint, filename='checkpoint.pth.tar'):
    torch.save(checkpoint, filename)


def get_checkpoint(collection, fs, checkpoint_name):
    checkpoint_exp = collection.find_one({'artifacts.name': checkpoint_name}, {'artifacts': 1})

    checkpoint_bin = None
    for checkpoint in checkpoint_exp['artifacts']:
        if checkpoint['name'] == checkpoint_name:
            assert checkpoint_bin is None
            checkpoint_bin = fs.get(checkpoint['file_id'])  # Gridout object

    assert checkpoint_bin is not None

    # Strange way to have the GridOut object as a python File
    temp_path = tempfile.mkdtemp()
    temp_file = os.path.join(temp_path, checkpoint_name)
    with open(temp_file, 'wb') as f:
        f.write(checkpoint_bin.read())
        chkpt = torch.load(temp_file)
    shutil.rmtree(temp_path)
    return chkpt


def format_exp_name(args, _run):
    exp_name = args['arch']
    if args['arch'] in ['ResCNF', 'FullResCNF', 'ResNet']:
        exp_name += '_bw' + str(args['blocks_width'])
        if args['bottlenecks']:
            exp_name += 'bn'

    if args['arch'] == "KWS":
        exp_name += '_{}-{}_l{}_s{}'.format(args['kws_model'], args['kws_ds'], args['frame_length'],
                                            args['frame_stride'])

    if args['arch'] == 'ResCNF' and isinstance(args['shift'], str):
        exp_name += args['shift']

    if args['arch'] in ['BSN']:
        exp_name += '_L' + str(args['layers']) + 'C' + str(args['channels'])

    if args['arch'] in ['3DCNF']:
        exp_name += '_L' + str(args['layers']) + 'B' + str(args['3d_blocks']) + 'C' + str(args['channels'])

    exp_name += '_' + args['dset'] + '_' + args['cost_optimization']

    if args['static'] > 0:
        exp_name += '_static_{}'.format(args['static'])
    else:
        exp_name += '_oc' + str(args['objective_cost']) + '_l' + str(args['lambda'])

    if args['exp_name'] != '':
        exp_name = args['exp_name'] + '_' + exp_name
    exp_name += '_' + str(_run._id)

    if args['test']:
        exp_name = 'TEST_' + exp_name

    return exp_name
