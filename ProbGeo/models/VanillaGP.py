import torch
import gpytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np 

class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        num_inducing = inducing_points.size(0)
        input_dim = inducing_points.size(1)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = (
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=input_dim-2,
                    active_dims=tuple(range(0, input_dim-2))
                )
            )
            +
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=2,
                    active_dims=tuple(range(input_dim - 2, input_dim))
                )
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseGaussianProcessRegressor:
    def __init__(
        self,
        num_inducing=500,
        lr=1e-2,
        num_epochs=100,
        device=None,
        y_mu=None,
        y_sigma=None
    ):
        self.num_inducing = num_inducing
        self.lr = lr
        self.num_epochs = num_epochs

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.likelihood = None
        self.y_mu = y_mu
        self.y_sigma = y_sigma
        self.history = []

    def _count_num_data(self, train_loader):
        num_data = 0

        for x_batch, y_batch in train_loader:
            num_data += x_batch.size(0)

        return num_data

    def _get_inducing_points(self, train_loader):
        inducing_points = []

        for x_batch, y_batch in train_loader:
            inducing_points.append(x_batch)

            current_n = sum(batch.size(0) for batch in inducing_points)

            if current_n >= self.num_inducing:
                break

        inducing_points = torch.cat(inducing_points, dim=0)
        inducing_points = inducing_points[:self.num_inducing]
        inducing_points = inducing_points.float().to(self.device)

        return inducing_points

    def _initialize_model(self, train_loader):
        inducing_points = self._get_inducing_points(train_loader)

        self.model = VariationalGPModel(inducing_points).to(self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

    def fit(self, train_loader, verbose=True):
        num_data = self._count_num_data(train_loader)

        if self.num_inducing > num_data:
            raise ValueError(
                f"num_inducing={self.num_inducing} cannot be larger than "
                f"the number of training samples={num_data}."
            )

        self._initialize_model(train_loader)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.lr
        )

        mll = gpytorch.mlls.VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=num_data
        )

        self.history = []

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)

                if y_batch.ndim > 1:
                    y_batch = y_batch.squeeze(-1)

                optimizer.zero_grad()

                output = self.model(x_batch)
                loss = -mll(output, y_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * x_batch.size(0)

            epoch_loss /= num_data
            self.history.append(epoch_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1:04d} | "
                    f"Negative ELBO: {epoch_loss:.4f}"
                )

        return self.history

    def predict(self, data_loader, return_std=True, return_targets=True):
        if self.model is None or self.likelihood is None:
            raise RuntimeError("You must fit the model before prediction.")

        self.model.eval()
        self.likelihood.eval()

        means = []
        variances = []
        targets = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.float().to(self.device)

                pred_dist = self.likelihood(self.model(x_batch))

                means.append(pred_dist.mean.cpu())
                variances.append(pred_dist.variance.cpu())

                if return_targets:
                    targets.append(y_batch.cpu())

        mean = torch.cat(means, dim=0)
        variance = torch.cat(variances, dim=0)

        if return_std:
            std = torch.sqrt(variance)

        if return_targets:
            y_true = torch.cat(targets, dim=0)

            if y_true.ndim > 1:
                y_true = y_true.squeeze(-1)

            if return_std:
                return mean, std, y_true

            return mean, y_true

        if return_std:
            return mean, std

        return mean

    def evaluate(self, data_loader, original_scale=True):
      if self.model is None or self.likelihood is None:
          raise RuntimeError("You must fit the model before evaluation.")

      self.model.eval()
      self.likelihood.eval()

      predictions = []
      targets = []

      total_nll = 0.0
      total_points = 0

      with torch.no_grad(), gpytorch.settings.fast_pred_var():
          for x_batch, y_batch in data_loader:
              x_batch = x_batch.float().to(self.device)
              y_batch = y_batch.float().to(self.device)

              if y_batch.ndim > 1:
                  y_batch = y_batch.squeeze(-1)

              pred_dist = self.likelihood(self.model(x_batch))

              mean_scaled = pred_dist.mean
              var_scaled = pred_dist.variance
              std_scaled = torch.sqrt(var_scaled.clamp_min(1e-9))

              normal_scaled = torch.distributions.Normal(
                  mean_scaled,
                  std_scaled
              )

              log_prob_scaled = normal_scaled.log_prob(y_batch)

              if original_scale and self.y_mu is not None and self.y_sigma is not None:
                  y_mu = self.y_mu.to(self.device).squeeze()
                  y_sigma = self.y_sigma.to(self.device).squeeze()

                  y_true_log = y_batch * y_sigma + y_mu
                  y_pred_log_mean = mean_scaled * y_sigma + y_mu
                  y_pred_log_var = var_scaled * y_sigma**2


                  log_prob_original = log_prob_scaled

                  batch_nll = -log_prob_original.sum()

                  predictions.append(y_pred_log_mean.cpu())
                  targets.append(y_true_log.cpu())

              else:
                  batch_nll = -log_prob_scaled.sum()

                  predictions.append(mean_scaled.cpu())
                  targets.append(y_batch.cpu())

              total_nll += batch_nll.item()
              total_points += y_batch.size(0)

      y_pred = torch.cat(predictions, dim=0).numpy()
      y_true = torch.cat(targets, dim=0).numpy()

      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      mae = mean_absolute_error(y_true, y_pred)
      r2 = r2_score(y_true, y_pred)
      nll = total_nll / total_points

      return {
          "R2": r2,
          "RMSE": rmse,
          "MAE": mae,
          "NLL": nll
      }
