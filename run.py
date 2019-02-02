#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 13e/01/2019
# @Author  : scrat
# @File    : run.py
# @Software: PyCharm
# @Desc    :
# @license : Copyright(C), benew


# import click
import fire
import os
import codecs
import yaml

from collider import run_class
from collider.utils.logger import system_log


# # @click.command()
# # @click.option("-n", "--flow_name", default=None)
# # @click.option("-f", "--configure_file", default=None)
# # @click.help_option("-h", "--help")
# def cli_run(**kwargs):
#     run(**kwargs)


def run(flow_name, config_file=None, **kwargs):
    # flow_name = kwargs['flow_name']
    # config_file = kwargs['configure_file']

    # if flow_name is None and config_file is None:
    #     raise AttributeError("Please input flow_name or configure_file")

    """
    当指定了flow_name的时候，指定的文件做配置, 再查找flow_name.yml作为配置，否则报错。
    当没有指定flow_name的时候，报错
    """

    if flow_name is not None:
        if config_file is None:
            new_config_file = os.path.join("configure", "%s.yml" % flow_name)
            if os.path.exists(new_config_file):
                system_log.warn("configure_file %s is override by %s" % (config_file, new_config_file))
                config_file = new_config_file
            else:
                raise FileNotFoundError("%s or %s both not found" % (config_file, new_config_file))
    else:
        raise AttributeError("please input flow_name")

    with codecs.open(config_file, encoding="utf-8") as f:
        config = yaml.load(f)

    m = __import__('flow.%s' % flow_name, fromlist=True)

    flow = getattr(m, flow_name)
    result = run_class(flow, config)
    print(result)


if __name__ == '__main__':
    run(flow_name="flow_attribution", configure_file=None)
    # cli_run()
    # fire.Fire(run)
