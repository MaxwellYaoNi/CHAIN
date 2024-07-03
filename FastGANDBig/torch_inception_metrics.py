import torch, numpy as np
from scipy import linalg
import os, pickle
import time
import torch.nn.functional as F


class InceptionWrapper(torch.nn.Module):
    def __init__(self, inception_model):
        super(InceptionWrapper, self).__init__()
        self._inception_model = inception_model

    def forward(self, x):
        features = self._inception_model(x, return_features=True)
        logits = F.linear(features, self._inception_model.output.weight)
        predictions = F.softmax(logits, 1)
        return features, predictions


class ImageWrapper:
    def __init__(self, images, labels=None):
        if images is not None:
            self._images = self._assert_images(images)
            self._images = torch.tensor(self._images)
            self._labels = torch.tensor(labels) if labels is not None else None

    def set_images_torch(self, images, labels=None):
        self._images = self._assert_images(images)
        self._labels = labels

    def set_None(self):
        self._images = None
        self._labels = None

    def _assert_images(self, images):
        assert (len(images[0].shape) == 3)
        assert (images[0].shape[0] == 3)
        assert (images[0].max() > 10)
        assert (images[0].min() >= 0.0)
        return images

    def sample(self, begin: int, end: int):
        return self._images[begin:end]

    def get_labels(self):
        return self._labels

    def set_labels(self, value=None):
        self._labels = value

    def __len__(self):
        return len(self._images)

class ImageH5Wrapper(ImageWrapper):
    def __init__(self, images, labels=None, img_len=None):
        super().__init__(None, None)
        self._images = images
        self._img_len = img_len
        self._labels = labels

    def __len__(self):
        return self._img_len

    def sample(self, begin: int, end: int):
        return torch.tensor(self._images[begin:end])

class ImageGeneratorWrapper(ImageWrapper):
    def __init__(self, sample_func, num_images):
        super().__init__(None, None)
        self.sample_func = sample_func
        self.num_images = num_images

    def __len__(self):
        return self.num_images

    def set_sample_func(self, sample_func):
        self.sample_func = sample_func

    def sample(self, begin: int, end: int):
        return self.sample_func(end-begin)




class TorchInceptionMetrics:
    """
       This class is designed totally using Torch for calculating
       IS(50k),
       train FID (50k fake images and real training data reference) and
       val FID (the same number of validation data and real validation data reference).

       Examples
       --------
       >>> inception_dir = '' #specify the inception model path
       >>> inception_metric = TorchInceptionMetrics(inception_dir)

       ### calculate Inception Score(IS)
       >>> train_data = None #specify the data to be evaluated, shape must be BCHW type, and values in [0, 255]
       ### calculate FID
       >>> real_moments_path = '' # specify the path of real moments
       >>> inception_metric.load_train_fid_moments(real_moments_path, train_data) # must call this function before calculate fid
       ### one can also specify the real data and use below to calculate real moments
       ### inception_metric.load_fid_real_moments(real_moments_path, real_data)


       ### simultaneously calculate IS and fid
       >>> IS_mean, IS_std, fid = inception_metric.get_IS_train_FID(train_data)

       ### for detailed usage, see _test() function
       >>> _test() #
       """

    def __init__(self, inception_path, num_gpus=None, batch_size=None, torch_for_fid=True, eps=1e-10):
        '''
        :param inception_path:
        :param num_gpus:
        :param batch_size:
        :param torch_for_fid: set to True if you have more than 2048 samples in your real dataset else False
        '''
        # torch.backends.cudnn.benchmark = True
        self._train_moments = None
        self._val_moments = None
        self._class_moments = None
        self._inception_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
        num_avail_gpus = torch.cuda.device_count()
        self._num_gpus = num_avail_gpus if num_gpus is None else min(num_avail_gpus, num_gpus)
        self._torch_for_scores = torch_for_fid
        self.eps = eps

        if batch_size is None:
            self._batch_size = 100 * max(self._num_gpus, 1)
            # if self._num_gpus <= 1:
            #     self._batch_size = 100
            # else:
            #     self._batch_size = 100 * self._num_gpus
        else:
            self._batch_size = batch_size

        # initialize model on cpu or multiple gpus
        self._model_loaded = False
        self._first_batch = False
        self._model_path = os.path.join(inception_path, 'inception-2015-12-05.pt')
        self._device = torch.device('cpu') if self._num_gpus == 0 else torch.device('cuda:0')

    def _load_model(self):
        if self._model_loaded:
            return
        fobj = open(self._model_path, 'rb')
        self._model = InceptionWrapper(torch.jit.load(fobj))
        fobj.close()
        self._device = torch.device('cpu')
        if self._num_gpus > 0:
            self._device = torch.device('cuda:0')
            self._model = self._model.to(self._device)
        if self._num_gpus > 1:
            self._model = torch.nn.DataParallel(self._model)
        self._model = self._model.eval().requires_grad_(False)
        self._model_loaded = True


    def _set_torch_for_scores(self, torch_for_scores):
        if self._torch_for_scores == torch_for_scores:
            return
        if self._torch_for_scores:
            if self._train_moments is not None:
                self._train_moments['mu'] = self._train_moments['mu'].cpu().numpy()
                self._train_moments['sigma'] = self._train_moments['sigma'].cpu().numpy()
            if self._val_moments is not None:
                self._val_moments['mu'] = self._val_moments['mu'].cpu().numpy()
                self._val_moments['sigma'] = self._val_moments['sigma'].cpu().numpy()
        else:
            if self._train_moments is not None:
                self._train_moments['mu'] = torch.from_numpy(self._train_moments['mu']).to(self._device)
                self._train_moments['sigma'] = torch.from_numpy(self._train_moments['sigma']).to(self._device)
            if self._val_moments is not None:
                self._val_moments['mu'] = torch.from_numpy(self._val_moments['mu']).to(self._device)
                self._val_moments['sigma'] = torch.from_numpy(self._val_moments['sigma']).to(self._device)
        self._torch_for_scores = torch_for_scores

    def _calculate_IS_numpy(self, predictions: np.array, splits: int = 10) -> tuple:
        scores = []
        split_len = predictions.shape[0] // splits
        start = 0
        for i in range(splits):
            end = start + split_len
            part = predictions[start:end, :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
            start = end
        return np.mean(scores).item(), np.std(scores).item()

    def _calculate_IS_torch(self, predictions: torch.FloatTensor, splits: int = 10) -> tuple:
        with torch.no_grad():
            split_len = predictions.shape[0] // splits
            predictions = predictions[:split_len * splits].reshape(splits, split_len, -1)
            kl = predictions * (torch.log(predictions) - torch.log(torch.mean(predictions, 1, keepdim=True)))
            kl = torch.mean(torch.sum(kl, -1), -1)
            score_std, score_mean = torch.std_mean(torch.exp(kl))
        return score_mean.cpu().item(), score_std.cpu().item()

    def _calculate_IS(self, predictions: np.array or torch.FloatTensor, splits: int = 10) -> tuple:
        '''
        tested on V100 with shape of predictions to be [50000, 1008]
        torch cpu time: 350ms
        torch gpu time: 1.3ms
        numpy cpu time: 189ms
        absolute difference between torch and numpy: score (2.4e-7), std(9.8e-7)
        :param predictions:
        :param splits:
        :return:
        '''
        if self._torch_for_scores:
            return self._calculate_IS_torch(predictions, splits)
        return self._calculate_IS_numpy(predictions, splits)

    def _calculate_moments_torch(self, features: torch.FloatTensor) -> tuple:
        with torch.no_grad():
            mu = torch.mean(features, 0)
            mu_features = features.type(torch.float64) - mu.type(torch.float64)
            sigma = 1.0 / (features.shape[0] - 1) * mu_features.t() @ mu_features
        return mu, sigma

    def _calculate_moments_numpy(self, features: np.ndarray) -> tuple:
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def _calculate_moments(self, features: torch.FloatTensor or np.ndarray) -> tuple:
        '''
        tested on V100 with shape of features to be [50000, 2048]
        torch cpu time: 3.02s
        torch gpu time: 0.44s
        numpy cpu time: 3.34s
        results differences(sum of abs across all elements) between torch and numpy: mu (0.0027) sigma (0.0017)
        :param features:
        :return:
        '''
        if self._torch_for_scores:
            return self._calculate_moments_torch(features)
        return self._calculate_moments_numpy(features)

    def _matrix_sqrt_newton_schulz(self, A: torch.Tensor, num_iters=50):
        '''
        for a PSD matrix with dims 2048x2048, num_iters=20 is already enough to obtain a good approximation.
        :param A: a PSD matrix, faster than matrix_sqrt_svd(),
         the algorithm may not converge if A isn't a PSD matrix.
        :param num_iters:
        :return:
        '''
        normA = torch.norm(A)
        Y = A / normA
        I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device, requires_grad=False)
        I3 = I * 3
        Z = torch.eye(A.shape[0], dtype=A.dtype, device=A.device, requires_grad=False)
        for _ in range(num_iters):
            T = 0.5 * (I3 - Z @ Y)
            Y = Y @ T
            Z = T @ Z
            # print(torch.sum(torch.abs(Y @ Z - I)).item()) # check if the algorithm converge,
        sqrt_normA = torch.sqrt(normA)
        # return sqrt_normA * Y, Z / sqrt_normA # matrix_sqrt, inverse_matrix_sqrt
        return sqrt_normA * Y

    def _calculate_frechet_distance_torch(self, mu_real, sigma_real, mu_fake, sigma_fake):
        with torch.no_grad():
            m = torch.square(mu_fake - mu_real).sum()
            # add a small disturb to prevent large values
            # when eps = 0: FID between CIFAR-10 train and val: 3.1508438728467922,
            # when eps=1e-10, FID between CIFAR-10 train and val: 3.1508417336644925
            # the ICFID between CIFAR-10 train and val become normal when eps >= 1e-11
            distance = m + torch.trace(sigma_real) + torch.trace(sigma_fake)
            if self.eps > 0:
                I_eps = torch.eye(sigma_fake.shape[0], device=sigma_fake.device) * self.eps
                sigma_fake = sigma_fake + I_eps
                sigma_real = sigma_real + I_eps
            s = self._matrix_sqrt_newton_schulz(sigma_fake @ sigma_real)
            distance -= 2 * s.trace()
        return distance.cpu().item()

    def _calculate_frechet_distance_numpy(self, mu_real, sigma_real, mu_fake, sigma_fake):
        """Numpy implementation of the Frechet Distance.
                The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
                and X_2 ~ N(mu_2, C_2) is
                        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
                Returns:
                --   : The Frechet Distance.
                """
        m = np.square(mu_fake - mu_real).sum()
        I_eps = np.eye(sigma_real.shape[0]) * self.eps
        sigma_real = sigma_real + I_eps
        sigma_fake = sigma_fake + I_eps
        s, _ = linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)  # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2 * s)
        return dist.item()

    def _calc_fid_robust(self, sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
        cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)
        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))
                return 99999.
                raise ValueError(f'Imaginary component {m}')
            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff
        trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)
        fid = mean_norm + trace
        return fid

    def _calculate_frechet_distance(self, mu_real, sigma_real, mu_fake, sigma_fake):
        '''
        example: the distance between train moments and val moments of CIFAR-10
        torch, gpu: distance: 3.150844544456845, time: 0.8642089366912842
        numpy, cpu: distance: 3.150844544450406, time: 6.260093927383423

        :param mu_real:
        :param sigma_real:
        :param mu_fake:
        :param sigma_fake:
        :return:
        '''
        if self._torch_for_scores:
            return self._calculate_frechet_distance_torch(mu_real, sigma_real, mu_fake, sigma_fake)
        return self._calculate_frechet_distance_numpy(mu_real, sigma_real, mu_fake, sigma_fake)

    def _calculate_moments_with_labels(self, features, labels):
        if labels is None:
            mu, sigma = self._calculate_moments(features)
        else:
            mus = []
            sigmas = []
            unsqueeze_op = torch.unsqueeze if self._torch_for_scores else np.expand_dims
            cat_op = torch.cat if self._torch_for_scores else np.concatenate
            for i in range(labels.max() + 1):
                mu_i, sigma_i = self._calculate_moments(features[labels == i])
                mus.append(unsqueeze_op(mu_i, 0))
                sigmas.append(unsqueeze_op(sigma_i, 0))
            mu = cat_op(mus)
            sigma = cat_op(sigmas)
        return mu, sigma

    def _load_fid_moments(self, moments_path: str, data: ImageWrapper = None, verbose: bool = False) -> dict:
        if os.path.exists(moments_path):
            moments_from_file = np.load(moments_path)
            moments = {'mu': moments_from_file['mu'], 'sigma': moments_from_file['sigma']}

            if verbose:
                print('Moments file exists at %s. Load moments successfully.' % (moments_path))
            if self._torch_for_scores:
                moments['mu'] = torch.from_numpy(moments['mu']).to(self._device)
                moments['sigma'] = torch.from_numpy(moments['sigma']).to(self._device)
        else:
            if verbose:
                print(
                    'Moments doesn\'t exist at %s. Moments will be calculated using given real data and will be stored at %s'
                    % (moments_path, moments_path))

            ## ensure the path exists
            os.makedirs(os.path.dirname(moments_path), exist_ok=True)

            features, _ = self._get_outputs(data, verbose)
            labels = data.get_labels()
            mu, sigma = self._calculate_moments_with_labels(features, labels)

            moments = {'mu': mu, 'sigma': sigma}
            if self._torch_for_scores:
                mu = mu.cpu().numpy()
                sigma = sigma.cpu().numpy()
            np.savez(moments_path, mu=mu, sigma=sigma)
        return moments


    def _get_outputs(self, image_wrapper: ImageWrapper, verbose: bool = False):
        self._load_model()
        start_time = time.time()
        all_features = []
        all_activations = []
        num_img = len(image_wrapper)
        with torch.no_grad():
            if not self._first_batch:
                # given the same input, the result of the first forward is not totally the same as later forwards
                # however, the second, third, forth ... forwards are the same.
                # this trick skip the first forward which has some randomness,
                batch_image = image_wrapper.sample(0, self._batch_size)
                if batch_image.device != self._device:
                    batch_image = batch_image.to(self._device)
                self._model(batch_image)
                self._first_batch = True

            for begin in range(0, num_img, self._batch_size):
                end = min(begin + self._batch_size, num_img)
                batch_image = image_wrapper.sample(begin, end)
                if batch_image.device != self._device:
                    batch_image = batch_image.to(self._device)
                features, activations = self._model(batch_image)
                all_features.append(features if self._torch_for_scores else features.cpu().numpy())
                all_activations.append(activations if self._torch_for_scores else activations.cpu().numpy())
                if verbose:
                    print('Inception Outputs: %d/%d, time cost:%d s' % (end, num_img, int(time.time() - start_time)))
        if self._torch_for_scores:
            return torch.cat(all_features), torch.cat(all_activations)
        return np.concatenate(all_features), np.concatenate(all_activations)

    def load_train_fid_moments(self, training_moments_path: str, training_data: ImageWrapper = None,
                               verbose: bool = False) -> None:
        """ Ensure the moments of training data to be loaded. if the moments file doesn't exist, moments for
            given training data will be calculation.
            Params:
            -- training_moments_path:   path of real moments needed for FID calculation.
            -- training_data        :   list or numpy arrays containing numpy arrays of dimension (3, hi, wi).
                                        The values must lie between 0 and 255. real_data can be set to None if
                                        real moments file exists
        """
        labels = None
        if training_data is not None:
            labels = training_data.get_labels()
            training_data.set_labels(None)
        self._train_moments = self._load_fid_moments(training_moments_path, training_data, verbose)
        if training_data is not None:
            training_data.set_labels(labels)

    def load_val_fid_moments(self, testing_moments_path: str, val_data: ImageWrapper = None,
                             verbose: bool = False) -> None:
        """ Ensure the moments of testing data to be loaded. if the moments file doesn't exist, moments for
            given training data will be calculation.
            Params:
            -- testing_moments_path:   path of real moments needed for FID calculation.
            -- testing_data        :   list or numpy arrays containing numpy arrays of dimension (hi, wi, 3).
                                       The values must lie between 0 and 255. real_data can be set to None if
                                       real moments file exists
        """
        labels = None
        if val_data is not None:
            labels = val_data.get_labels()
            val_data.set_labels(None)
        self._val_moments = self._load_fid_moments(testing_moments_path, val_data, verbose)
        if val_data is not None:
            val_data.set_labels(labels)

    def load_class_fid_moments(self, class_moments_path: str, class_data: ImageWrapper = None,
                             verbose: bool = False) -> None:
        """ Ensure the moments of testing data to be loaded. if the moments file doesn't exist, moments for
            given training data will be calculation.
            Params:
            -- testing_moments_path:   path of real moments needed for FID calculation.
            -- testing_data        :   list or numpy arrays containing numpy arrays of dimension (hi, wi, 3).
                                       The values must lie between 0 and 255. real_data can be set to None if
                                       real moments file exists
        """
        if class_data is not None:
            assert class_data.get_labels() is not None
        self._class_moments = self._load_fid_moments(class_moments_path, class_data, verbose)

    def get_IS_train_FID(self, image_wrapper: ImageWrapper = None, verbose: bool = False):
        if self._train_moments is None:
            raise Exception(
                'Moments of training data aren\'t given, '
                'please call load_train_fid_moments() before call get_IS_train_FID()')
        features, predictions = self._get_outputs(image_wrapper, verbose=verbose)
        IS_mean, IS_std = self._calculate_IS(predictions)
        mu, sigma = self._calculate_moments(features)
        tFID = self._calculate_frechet_distance(self._train_moments['mu'], self._train_moments['sigma'], mu, sigma)
        return IS_mean, IS_std, tFID

    def get_IS_FIDs(self, image_wrapper: ImageWrapper, num_val_data: int = None, verbose: bool = False):
        '''
        #remark: for both official tensorflow and torch implementations, the results are varied across different batch_size and number of GPUs.
        torch_for_scores=True, CIFAR-10 training data:
            time: 50.12s(1GPU), 18.94s(4GPUs)
            IS_mean: 11.2373685836792   (abs difference compared to tensorflow implementation: 2e-6)
            IS_std: 0.1225217804312706  (abs difference: 6e-3)
            tFID: 1.512034941697493e-11 (abs difference: 4e-8)
            vFID: 3.150843884574158     (abs difference: 2e-6)
        torch_for_scores=False, CIFAR-10 training data:
            time: 63.09s(1GPU),
            IS_mean: 11.237369537353516 (abs difference compared to tensorflow implementation: 3e-6)
            IS_std: 0.11623427271842957 (abs difference: 5e-7)
            tFID: 3.811399112985114e-12 (abs difference: 4e-8)
            vFID: 3.1508436983082557    (abs difference: 2e-6)
        original tensorflow implementations, CIFAR-10 training data:
            time: 94.88s(1GPU), 37.11(4GPUs)
            IS_mean: 11.237367234146854
            IS_std: 0.11622645638305688
            tFID: 4.095829409496217e-08
            vFID: 3.150845646947172
        '''
        if self._train_moments is None:
            raise Exception(
                'Moments of training data aren\'t given, '
                'please call load_train_fid_moments() before call get_IS_and_FIDs()')
        if self._val_moments is None:
            raise Exception(
                'Moments of testing data aren\'t given, '
                'please call load_val_fid_moments() before call get_IS_and_FIDs()')

        features, predictions = self._get_outputs(image_wrapper, verbose)

        IS_mean, IS_std = self._calculate_IS(predictions)
        mu, sigma = self._calculate_moments(features)
        tFID = self._calculate_frechet_distance(self._train_moments['mu'], self._train_moments['sigma'], mu, sigma)

        if num_val_data is None or num_val_data == len(image_wrapper):
            val_mu, val_sigma = mu, sigma
        else:
            val_mu, val_sigma = self._calculate_moments(features[-num_val_data:])
        vFID = self._calculate_frechet_distance(self._val_moments['mu'], self._val_moments['sigma'], val_mu, val_sigma)
        return IS_mean, IS_std, tFID, vFID, []

    def get_all_metrics(self, image_wrapper: ImageWrapper, num_val_data: int = None, verbose: bool = False):
        '''
        #remark: for both official tensorflow and torch implementations, the results are varied across different batch_size and number of GPUs.
        torch_for_scores=True, CIFAR-10 training data:
            time: 50.12s(1GPU), 18.94s(4GPUs)
            IS_mean: 11.2373685836792   (abs difference compared to tensorflow implementation: 2e-6)
            IS_std: 0.1225217804312706  (abs difference: 6e-3)
            tFID: 1.512034941697493e-11 (abs difference: 4e-8)
            vFID: 3.150843884574158     (abs difference: 2e-6)
        torch_for_scores=False, CIFAR-10 training data:
            time: 63.09s(1GPU),
            IS_mean: 11.237369537353516 (abs difference compared to tensorflow implementation: 3e-6)
            IS_std: 0.11623427271842957 (abs difference: 5e-7)
            tFID: 3.811399112985114e-12 (abs difference: 4e-8)
            vFID: 3.1508436983082557    (abs difference: 2e-6)
        original tensorflow implementations, CIFAR-10 training data:
            time: 94.88s(1GPU), 37.11(4GPUs)
            IS_mean: 11.237367234146854
            IS_std: 0.11622645638305688
            tFID: 4.095829409496217e-08
            vFID: 3.150845646947172
        '''
        if self._train_moments is None:
            raise Exception(
                'Moments of training data aren\'t given, '
                'please call load_train_fid_moments() before call get_IS_and_FIDs()')
        if self._val_moments is None:
            raise Exception(
                'Moments of testing data aren\'t given, '
                'please call load_val_fid_moments() before call get_IS_and_FIDs()')

        features, predictions = self._get_outputs(image_wrapper, verbose)

        IS_mean, IS_std = self._calculate_IS(predictions)
        mu, sigma = self._calculate_moments(features)
        tFID = self._calculate_frechet_distance(self._train_moments['mu'], self._train_moments['sigma'], mu, sigma)

        if num_val_data is None or num_val_data == len(image_wrapper):
            val_mu, val_sigma = mu, sigma
        else:
            val_mu, val_sigma = self._calculate_moments(features[-num_val_data:])
        vFID = self._calculate_frechet_distance(self._val_moments['mu'], self._val_moments['sigma'], val_mu, val_sigma)

        IS_10k, _ = self._calculate_IS(predictions[-10000:])
        mu_10k, sigma_10k = self._calculate_moments(features[-10000:])
        FID_10k = self._calculate_frechet_distance(self._train_moments['mu'], self._train_moments['sigma'], mu_10k, sigma_10k)
        return IS_mean, IS_std, tFID, vFID, [IS_10k, FID_10k]

    def get_IS_ICFIDs(self, image_wrapper: ImageWrapper, num_val_data: int = None, verbose: bool = False):
        if self._train_moments is None:
            raise Exception(
                'Moments of training data aren\'t given, '
                'please call load_train_fid_moments() before call get_IS_and_FIDs()')

        features, predictions = self._get_outputs(image_wrapper, verbose)

        IS_mean, IS_std = self._calculate_IS(predictions)
        mu, sigma = self._calculate_moments(features)
        tFID = self._calculate_frechet_distance(self._train_moments['mu'], self._train_moments['sigma'], mu, sigma)
        if num_val_data is None or num_val_data == len(image_wrapper):
            val_mu, val_sigma = mu, sigma
        else:
            val_mu, val_sigma = self._calculate_moments(features[-num_val_data:])
        vFID = self._calculate_frechet_distance(self._val_moments['mu'], self._val_moments['sigma'], val_mu, val_sigma)

        cls_mu, cls_sigma = self._calculate_moments_with_labels(features, image_wrapper.get_labels())
        cFIDs = []
        for i in range(cls_mu.shape[0]):
            cFIDs.append(self._calculate_frechet_distance(self._class_moments['mu'][i], self._class_moments['sigma'][i],
                                                          cls_mu[i], cls_sigma[i]))

        return IS_mean, IS_std, tFID, vFID, cFIDs

    def get_IS_tFID(self, image_wrapper: ImageWrapper, verbose: bool = False):
        if self._train_moments is None:
            raise Exception(
                'Moments of training data aren\'t given, '
                'please call load_train_fid_moments() before call get_IS_and_FIDs()')

        features, predictions = self._get_outputs(image_wrapper, verbose)

        IS_mean, IS_std = self._calculate_IS(predictions)
        mu, sigma = self._calculate_moments(features)
        tFID = self._calculate_frechet_distance(self._train_moments['mu'], self._train_moments['sigma'], mu, sigma)
        return IS_mean, IS_std, tFID

    def sample_and_FID(self, generator_func, num_gen_data=5000, verbose=False):
        if self._train_moments is None:
            raise Exception(
                'Moments of training data aren\'t given, '
                'please call load_train_fid_moments() before call get_IS_and_FIDs()')

        def sample_func(batch_images):
            all_images = []
            num_samples = 0
            with torch.no_grad():
                while num_samples < batch_images:
                    samples = generator_func()
                    num_samples += samples.shape[0]
                    all_images.append(samples.mul(255).clamp(0, 255).detach().int())
                all_images = torch.cat(all_images)[:batch_images]
            return all_images

        image_wrapper = ImageGeneratorWrapper(sample_func, num_images=num_gen_data)
        features, predictions = self._get_outputs(image_wrapper, verbose=verbose)
        mu, sigma = self._calculate_moments(features)
        FID = self._calculate_frechet_distance(self._train_moments['mu'], self._train_moments['sigma'], mu, sigma)
        return FID



