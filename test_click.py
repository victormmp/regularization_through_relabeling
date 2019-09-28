# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:39:02 2019

@author: victo
"""

import click
import time

with click.progressbar(range(100)) as bar:
    for i in bar:
        time.sleep(0.05)
    print('finished')