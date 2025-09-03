%% ------------------------------------------------------------------------
clear

%% SECTION 1: Load data
maindata  = readtable('../data/census2000.csv');
data_table = maindata(:, {'edu', 'edu_sp'});

% 5x5 education contingency
[contingency_matrix, ~, ~] = crosstab(maindata.edu, maindata.edu_sp);
L_data = contingency_matrix / sum(contingency_matrix,"all");

%% Suppose you have L_data from a 5x5 contingency table
M = 5; 
F = 5;

%% SECTION 2: Generate random errors and estimate alpha, sigma
R = 100;
R_final = R;
MU_accumulate = zeros(M,F);
mu_r         = NaN(M,F,R_final);
mu_zeroalpha = NaN(M,F,R_final);
E_mat_r      = NaN(M,F,R_final);
errors       = NaN(M,F,R);

k = 0;
for m = 1:M
    for f_ = 1:F
        k = k + 1;
        rng(k)
        u  = rand(R/2,1);
        au = 1 - u;
        errors(m,f_,:) = norminv([u; au]);
    end
end

[alpha_hat, sigma_hat] = fitDGP_exponentErrorDistOneVar(M, F, L_data, errors);
fprintf('alpha = %.4f, sigma = %.4f (one-variance MSM)\n', alpha_hat, sigma_hat);

% fix sigma for the residual maps used below
sigma_fix = 0.1;
k = 0;
for m = 1:M
    for f_ = 1:F
        k = k + 1;
        E_mat_r(m,f_,:) = sigma_fix * errors(m,f_,:);
    end
end

%% SECTION 3: Build final mu's over R draws
for r = 1:R_final
    mu_r(:,:,r)         = buildMuFromAlphaA(M, F, alpha_hat, E_mat_r(:,:,r));
    mu_zeroalpha(:,:,r) = buildMuFromAlphaA(M, F, 0,         E_mat_r(:,:,r));
    MU_accumulate       = MU_accumulate + mu_r(:,:,r);
end
mu_final = MU_accumulate / R_final;

sum_logodd_final = sum(logoddgen(mu_final), "all");
sum_logodd_data  = sum(logoddgen(L_data),   "all");
disp(sum_logodd_final);  disp(sum_logodd_data);

for r = 1:R_final
    T_dis(r,1)  = sum(logoddgen(mu_r(:,:,r)), "all");
    T_zero(r,1) = sum(logoddgen(mu_zeroalpha(:,:,r)), "all");
end

%% SECTION 4: Generate synthetic data for different alpha, N
alpha_vec = [0, 0.1, alpha_hat*0.5, alpha_hat];
n_vec     = [500, 2000, 120000];
sz_alpha  = numel(alpha_vec);
sz_n      = numel(n_vec);

%% SECTION 5: Simulations under weaker alpha
param.M=5; param.F=5;

pval_TP2        = NaN(R,sz_alpha,sz_n);
pval_DP2        = NaN(R,sz_alpha,sz_n);
LR_TP2          = NaN(R,sz_alpha,sz_n);
LR_DP2          = NaN(R,sz_alpha,sz_n);
contingency_full= NaN(M,F,R,sz_alpha,sz_n);

% --- NEW: holders for SK bootstrap
wald_T      = NaN(R,sz_alpha,sz_n);
wald_p      = NaN(R,sz_alpha,sz_n);
lr_T        = NaN(R,sz_alpha,sz_n);
lr_p        = NaN(R,sz_alpha,sz_n);

% bootstrap reps for SK tests
B_SK = 1000;

for a = 1:sz_alpha
    alpha_prime = alpha_vec(a);
    for k = 1:sz_n
        N = n_vec(k);
        for r = 1:R
            % 1) simulate table under alpha'
            mu_weaker   = buildMuFromAlphaA(M, F, alpha_prime, E_mat_r(:,:,r));
            truesimul   = generateTableWithEdu(mu_weaker, N);
            contingency_r = myTabulate(truesimul.edu, truesimul.edu_sp, M, F);
            % (NO forcing ones; zeros are allowed)
            contingency_full(:,:,r,a,k) = contingency_r;

            % 2) Siow tests + bootstrap distributions (TP2/DP2)
            [lr_uncon, lr_tp2, lr_dp2, mu_tp2, mu_dp2] = L_siow(contingency_r, param, 1);
            LR_TP2(r,a,k) = 2*(lr_uncon - lr_tp2);
            LR_DP2(r,a,k) = 2*(lr_uncon - lr_dp2);

            % For each r,a,k we generate bootstrap distributions once
            % (small B inside r-loop to keep memory bounded)
            B = 200;  % you can raise to 1000+ if runtime permits
            LR_TP2_dis = zeros(B,1);
            LR_DP2_dis = zeros(B,1);

            parfor b = 1:B
                % TP2 world
                boot_tp2      = generateTableWithEdu(mu_tp2/N, N);
                match_b_tp2   = crosstab(boot_tp2.edu, boot_tp2.edu_sp);
                [lu_b_tp2, lp_b_tp2, ~,~,~] = L_siow(match_b_tp2, param, 2);
                LR_TP2_dis(b) = 2*(lu_b_tp2 - lp_b_tp2);

                % DP2 world
                boot_dp2      = generateTableWithEdu(mu_dp2/N, N);
                match_b_dp2   = crosstab(boot_dp2.edu, boot_dp2.edu_sp);
                [lu_b_dp2, ~, lp_b_dp2,~,~] = L_siow(match_b_dp2, param, 3);
                LR_DP2_dis(b) = 2*(lu_b_dp2 - lp_b_dp2);
            end

            pval_TP2(r,a,k) = (1 + sum(LR_TP2_dis >= LR_TP2(r,a,k))) / (B + 1);
            pval_DP2(r,a,k) = (1 + sum(LR_DP2_dis >= LR_DP2(r,a,k))) / (B + 1);

            % 3) NEW: SK bootstrap (random-matching null) for this table
            SK = bootstrap_SK(contingency_r, M, F, B_SK, 1000 + 17*r + 31*a + 7*k);
            wald_T(r,a,k) = SK.wald_T;   wald_p(r,a,k) = SK.wald_p;
            lr_T(r,a,k)   = SK.lr_T;     lr_p(r,a,k)   = SK.lr_p;
        end
    end
end

% Averages over R for display
average_TP2       = squeeze(mean(LR_TP2,1,'omitnan'));
average_DP2       = squeeze(mean(LR_DP2,1,'omitnan'));
average_pval_TP2  = squeeze(mean(pval_TP2,1,'omitnan'));
average_pval_DP2  = squeeze(mean(pval_DP2,1,'omitnan'));

average_wald      = squeeze(mean(wald_T,1,'omitnan'));
average_pWald     = squeeze(mean(wald_p,1,'omitnan'));
average_LR_SK     = squeeze(mean(lr_T,1,'omitnan'));
average_pValSK    = squeeze(mean(lr_p,1,'omitnan'));

%% === LaTeX tables (use bootstrap p-values) ===
createLatexTableWithStars(alpha_vec, n_vec, average_wald',  average_pWald')
createLatexTableWithStars(alpha_vec, n_vec, average_TP2',   average_pval_TP2')
createLatexTableWithStars(alpha_vec, n_vec, average_DP2',   average_pval_DP2')
createLatexTableWithStars(alpha_vec, n_vec, average_LR_SK', average_pValSK')
