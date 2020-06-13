from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        # self.writer = tf.summary.create_file_writer(log_dir)
        self.writer = SummaryWriter()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag,value,step)

# {'class_conf': {'train':4,
#                   'val':2.3000,},
#  'class_blah': {'train':4,
#                   'val':2.3000,}}

    def list_of_scalars_summary(self, tag_value_list, step):
        """Log scalar variables."""
        # For each metric in each layer -> val and train loss on a single graph
        for tag, value in tag_value_list.items():
            self.writer.add_scalars(tag,value,step)
       
    # def __del__(self): 
    #     self.writer.close()