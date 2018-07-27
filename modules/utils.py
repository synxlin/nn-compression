def get_sparsity(param):
    """

    :param param:
    :return:
    """
    mask = param.eq(0)
    return float(mask.sum()) / mask.numel()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class StageScheduler(object):
    """

    """

    def __init__(self, max_num_stage, stage_step=45):
        """

        :param max_num_stage:
        :param stage_step:
        """
        self.max_num_stage = max_num_stage

        self.stage_step = stage_step
        if isinstance(stage_step, int):
            self.stage_step = [stage_step] * max_num_stage
        if isinstance(stage_step, str):
            self.stage_step = list(map(int, stage_step.split(',')))
        assert isinstance(self.stage_step, list)

        num_stage = len(self.stage_step)
        if num_stage < self.max_num_stage:
            for i in range(self.max_num_stage - num_stage):
                self.stage_step.append(self.stage_step[num_stage - 1])
        elif num_stage > self.max_num_stage:
            self.max_num_stage = num_stage
        assert len(self.stage_step) == self.max_num_stage

        for i in range(1, self.max_num_stage):
            self.stage_step[i] += self.stage_step[i - 1]

    def step(self, epoch):
        """

        :param epoch:
        :return:
        """
        stage = self.max_num_stage - 1
        for i, max_epoch in enumerate(self.stage_step):
            if epoch < max_epoch:
                stage = i
                break
        if stage > 0:
            epoch -= self.stage_step[stage - 1]
        return stage, epoch
