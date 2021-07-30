from __future__ import division, print_function, absolute_import
import os
import time

class RegressionLog:
    def __init__(self, filename):
        if os.path.exists(filename):
            self.output_file = open(filename, 'a')
        else:
            self.output_file = open(filename, 'w')
            # write a header
            self.output_file.write('epoch#,epoch_loss,epoch_r2,val_loss,val_r2,timestamp\n')

    def log(self, epoch, epoch_loss, epoch_r2, val_loss, val_r2):
        timestamp = time.strftime('%Y-%m-%d %H:%M')

        self.output_file.write(','.join(map(str, [epoch, epoch_loss, epoch_r2, 
                                        val_loss, val_r2, timestamp])) + '\n')
        self.output_file.flush()

    def log_result(self, epoch, epoch_results, val_results=None):
        if val_results is None:
            self.log(epoch, epoch_results.cost, epoch_results.rsquared, '', '')
        else:
            self.log(epoch, epoch_results.cost, epoch_results.rsquared, val_results.cost, val_results.rsquared)

class ClassificationAutoencoderLog:
    def __init__(self, filename):
        if os.path.exists(filename):
            self.output_file = open(filename, 'a')
        else:
            self.output_file = open(filename, 'w')
            # write a header
            self.output_file.write('epoch#,epoch_loss,epoch_recon,epoch_f1,val_loss,val_recon,val_f1,timestamp\n')

    def log(self, epoch, epoch_loss, epoch_recon, epoch_f1, val_loss, val_recon, val_f1):
        timestamp = time.strftime('%Y-%m-%d %H:%M')

        self.output_file.write(','.join(map(str, 
            [epoch, epoch_loss, epoch_recon, epoch_f1,
            val_loss, val_recon, val_f1, timestamp])) + '\n')
        self.output_file.flush()

    def log_result(self, epoch, epoch_results, val_results=None):
        if val_results is None:
            self.log(epoch, 
                epoch_results.cost, epoch_results.recon_error(), epoch_results.f1(), 
                '', '', '')
        else:
            self.log(epoch, 
                epoch_results.cost, epoch_results.recon_error(), epoch_results.f1(),
                val_results.cost, val_results.recon_error(), val_results.f1())

class CenterLossAELog:
    def __init__(self, filename):
        if os.path.exists(filename):
            self.output_file = open(filename, 'a')
        else:
            self.output_file = open(filename, 'w')
            # write a header
            self.output_file.write('epoch#,epoch_loss,epoch_recon,epoch_f1,epoch_cl,val_loss,val_recon,val_f1,val_cl,timestamp\n')

    def log(self, epoch, epoch_loss, epoch_recon, epoch_f1, epoch_cl, val_loss, val_recon, val_f1, val_cl):
        timestamp = time.strftime('%Y-%m-%d %H:%M')

        self.output_file.write(','.join(map(str, 
            [epoch, epoch_loss, epoch_recon, epoch_f1, epoch_cl,
            val_loss, val_recon, val_f1, val_cl, timestamp])) + '\n')
        self.output_file.flush()

    def log_result(self, epoch, epoch_results, val_results=None):
        if val_results is None:
            self.log(epoch, 
                epoch_results.cost, epoch_results.recon_error(), epoch_results.f1(), epoch_results.center_loss, 
                '', '', '', '')
        else:
            self.log(epoch, 
                epoch_results.cost, epoch_results.recon_error(), epoch_results.f1(), epoch_results.center_loss, 
                val_results.cost, val_results.recon_error(), val_results.f1(), epoch_results.center_loss)


class String_Log:
    def __init__(self, filename):
        if os.path.exists(filename):
            self.output_file = open(filename, 'a')
        else:
            self.output_file = open(filename, 'w')

    def log(self, message):
        self.output_file.write(message+'\n')
        self.output_file.flush()

class ClassifierLog:
    def __init__(self, filename):
        if os.path.exists(filename):
            self.output_file = open(filename, 'a')
        else:
            self.output_file = open(filename, 'w')
            # write a header
            self.output_file.write( \
                'epoch#,epoch_loss,epoch_f1,val_loss,val_f1,timestamp\n')

    def log(self, epoch, epoch_loss, epoch_f1, val_loss, val_f1):
        timestamp = time.strftime('%Y-%m-%d %H:%M')

        self.output_file.write(','.join(map(str, [epoch, epoch_loss, epoch_f1, 
                                        val_loss, val_f1, timestamp])) + '\n')
        self.output_file.flush()

    def log_result(self, epoch, epoch_results, val_results=None):
        if val_results is None:
            self.log(epoch, epoch_results.cost, epoch_results.f1(), '', '')
        else:
            self.log(epoch, 
                epoch_results.cost, epoch_results.f1(), 
                val_results.cost, val_results.f1())

class AELog:
    def __init__(self, filename):
        if os.path.exists(filename):
            self.output_file = open(filename, 'a')
        else:
            self.output_file = open(filename, 'w')
            # write a header
            self.output_file.write('epoch#,epoch_loss,val_loss,timestamp\n')

    def log(self, epoch, epoch_loss, val_loss):
        timestamp = time.strftime('%Y-%m-%d %H:%M')

        self.output_file.write(','.join(map(str, 
            [epoch, epoch_loss, val_loss, timestamp])) + '\n')
        self.output_file.flush()

    def log_result(self, epoch, epoch_results, val_results=None):
        if val_results is None:
            self.log(epoch, epoch_results.cost, '')
        else:
            self.log(epoch, epoch_results.cost, val_results.cost)

