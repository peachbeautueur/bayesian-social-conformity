data {
  int<lower=1> N;
  array[N] int<lower=1, upper=8> first_rating;
  array[N] int<lower=1, upper=8> group_rating;
  vector[N] second_rating;
  real<lower=0> alpha0;
  real<lower=0> beta0;
  real<lower=0> n_direct;
  real<lower=0> n_social;
}

transformed data {
  vector[N] p_direct;
  vector[N] p_social;
  vector[N] k_direct;
  vector[N] k_social;

  for (n in 1:N) {
    // Map 1-8 ratings to probabilities in [0, 1].
    p_direct[n] = (first_rating[n] - 1.0) / 7.0;
    p_social[n] = (group_rating[n] - 1.0) / 7.0;

    // Convert mapped probabilities into pseudo-count evidence.
    k_direct[n] = p_direct[n] * n_direct;
    k_social[n] = p_social[n] * n_social;
  }
}

parameters {
  real<lower=0, upper=5> w_direct;
  real<lower=0, upper=5> w_social;
  real<lower=1e-6, upper=2> sigma;
}

transformed parameters {
  vector[N] mu;

  for (n in 1:N) {
    real alpha_post;
    real beta_post;
    real posterior_mean;

    alpha_post = alpha0 + w_direct * k_direct[n] + w_social * k_social[n];
    beta_post =
      beta0
      + w_direct * (n_direct - k_direct[n])
      + w_social * (n_social - k_social[n]);

    posterior_mean = alpha_post / (alpha_post + beta_post);
    mu[n] = 1.0 + 7.0 * posterior_mean;
  }
}

model {
  // Weakly informative priors consistent with the constrained parameter ranges.
  w_direct ~ normal(1, 1);
  w_social ~ normal(1, 1);
  sigma ~ normal(0.5, 0.5);

  second_rating ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;

  for (n in 1:N) {
    log_lik[n] = normal_lpdf(second_rating[n] | mu[n], sigma);
    y_rep[n] = normal_rng(mu[n], sigma);
  }
}
