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
  vector[N] mu;

  for (n in 1:N) {
    real alpha_post;
    real beta_post;
    real posterior_mean;

    // Map 1-8 ratings to probabilities in [0, 1].
    p_direct[n] = (first_rating[n] - 1.0) / 7.0;
    p_social[n] = (group_rating[n] - 1.0) / 7.0;

    // Convert mapped probabilities into pseudo-count evidence.
    k_direct[n] = p_direct[n] * n_direct;
    k_social[n] = p_social[n] * n_social;

    // SBA takes both evidence sources at face value.
    alpha_post = alpha0 + k_direct[n] + k_social[n];
    beta_post = beta0 + (n_direct - k_direct[n]) + (n_social - k_social[n]);
    posterior_mean = alpha_post / (alpha_post + beta_post);

    // Map the posterior mean back to the 1-8 rating scale.
    mu[n] = 1.0 + 7.0 * posterior_mean;
  }
}

parameters {
  real<lower=1e-6, upper=2> sigma;
}

model {
  // Mildly regularizing prior centered near the synthetic observation scale.
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
