
%% ------------------------------------------------------------------------

clear

%% SECTION 1: Load data
maindata = readtable('../data/census2000.csv');
var_names = maindata.Properties.VariableNames;
data_table = maindata(:, {'edu', 'edu_sp'});

% Extract categorical data and build contingency table
[contingency_matrix, edu_groups, edu_sp_groups] = crosstab(maindata.edu, maindata.edu_sp);
L_data = contingency_matrix / sum(contingency_matrix,"all");

%% ------------------------------------------------------------------------
%{
==========================================
        Step 1: Fitting the model
==========================================
1) We  fit the DGP:
2) The objective is minimize the matching table
3) We solve for alpha >= 0 and eps_{m,w}, subject to e_{1,1}=0 for ID.
4) Then we reduce alpha -> alpha' < alpha, re-use eps_{m,w},
   and get a "weaker" distribution.
%}
%% ------------------------------------------------------------------------

%{
%--------------------------%
% A) Setup (commented-out example)
%--------------------------%
% M = 5;
% F = 5;
% 
% % The actual log-odds data (4x4) to be fitted
% L_data = contingency_matrix;
% L_data = L_data / sum(L_data,"all");

%--------------------------%
% B) Fit the model (commented-out example)
%--------------------------%
% fprintf('\n=== Fit the Model (log-odds SSE) ===\n');
% 
% % We'll solve for param = [ alpha, eps_1, eps_2, ..., eps_(M*F-1) ]
% % where we fix eps_{1,1} = 0 for identification.
% [alpha_hat, E_mat_hat] = fitDGP_exponentError(M, F, L_data);
% 
% fprintf('Estimated alpha = %.4f\n', alpha_hat);
%}
% -------------------------------------------------------------------------

%% Suppose you have L_data from a 5x5 contingency table
M = 5; 
F = 5;

% If you want a final "fitted distribution" using these estimates, we'll
% proceed with the (placeholder) fit below.

%% SECTION 2: Generate random errors and estimate alpha, sigma
R = 100;
R_final = R;
MU_accumulate = zeros(M,F);
mu_r = NaN(M,F,R_final);
mu_zeroalpha = NaN(M,F,R_final);
E_mat_r = NaN(5,5,R_final);
errors = NaN(5,5,R);

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

% Estimate using one sigma
[alpha_hat, sigma_hat] = fitDGP_exponentErrorDistOneVar(M, F, L_data, errors);

% (Commented-out alternative)
% [alpha_hat, sigma_hat] = fitDGP_exponentErrorDistManySigs(M, F, L_data, errors);

fprintf('alpha = %.4f, sigma = %.4f (one-variance MSM)\n', alpha_hat, sigma_hat);

%% Fix sigma to 0.1 for demonstration
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

sum_logodd_final   = sum(logoddgen(mu_final), "all");
sum_logodd_data    = sum(logoddgen(L_data),   "all");

% Print to verify
disp(sum_logodd_final)
disp(sum_logodd_data)

sigma_map = NaN(M,F);
% k=0
% for m=1:M
%     for f=1:F
%         k=k+1;
%         sigma_map(m,f)=sigma_hat(k);
%     end
% end

for r = 1:R_final
    T_dis(r,1)  = sum(logoddgen(mu_r(:,:,r)), "all");
    T_zero(r,1) = sum(logoddgen(mu_zeroalpha(:,:,r)), "all");
end

% Construct the fitted distribution:
% mu_hat = buildMuFromAlphaA(M, F, alpha_hat, E_mat_hat);

% Compare fitted log-odds to L_data
L_hat  = logoddgen(mu_final);
% sse_fit = sum((L_hat(:) - L_data(:)).^2);
% fprintf('SSE in log-odds space = %.6f\n', sse_fit);

%% SECTION 4: Generate synthetic data for different alpha, N
alpha_vec = [0, 0.1, alpha_hat*0.5, alpha_hat];
n_vec     = [500, 2000, 120000];
sz_alpha  = numel(alpha_vec);
sz_n      = numel(n_vec);

%{
==========================================
        Step 2:
==========================================
1) Take the "true" population matching vector for weakened alpha
2) Simulate N observations of multinomial random variables using the
   matching distribution
%}

%% SECTION 5: Simulations under weaker alpha
method           = 1; % Chi-square (T^2/s^2 )
sigma_hat        = 1;
T1s1_stat        = NaN(R,sz_alpha,sz_n);
pval_TP2       = NaN(R,sz_alpha,sz_n);
pval_DP2        = NaN(R,sz_alpha,sz_n);
s_data           = NaN(M,F,R,sz_alpha,sz_n);
contingency_full = NaN(M,F,R,sz_alpha,sz_n);
p_values         = NaN(R,sz_alpha,sz_n);
average_pvalues  = NaN(sz_alpha,sz_n);
B=10;
LR_TP2=NaN(R,sz_alpha,sz_n);
LR_DP2=NaN(R,sz_alpha,sz_n);
% N=size(truesimul,1);
param.M=5;
param.F=5;


for a = 1:sz_alpha
    alpha_prime = alpha_vec(a);
    for k = 1:sz_n
        for r = 1:R
            mu_weaker = buildMuFromAlphaA(M, F, alpha_prime, E_mat_r(:,:,r)); %Build matching table given alpha and the residual structure E_mat
            truesimul = generateTableWithEdu(mu_weaker, n_vec(k)); %Simulated Individual Data given the matching table mu and size n
            contingency_r = myTabulate(truesimul.edu, truesimul.edu_sp, M, F); % Simulated Matching Data 
            contingency_r(contingency_r==0) = 1;   % Prevent empty cells
            contingency_full(:,:,r,a,k) = contingency_r; 
            
            % L_siow
            % Input: Matching frequency table (not the cells)
            % Output: LR_uncon, LR_tp, LR_dp, mu_tp, mu_dp

                [lr_uncon, lr_tp2, lr_dp2,mu_tp2,mu_dp2] = L_siow(contingency_r, param,1);
                simulwith_TP2=generateTableWithEdu(mu_tp2/n_vec(k),n_vec(k));
                simulwith_DP2=generateTableWithEdu(mu_dp2/n_vec(k),n_vec(k));
                       LR_TP2(r,a,k)=2*(lr_uncon-lr_tp2);   
                       LR_DP2(r,a,k)=2*(lr_uncon-lr_dp2);          
                            N=n_vec(k);
                                %Bootstrap Simulation: Generate
                                %Distribution of TP2 and DP2 For a given
                                %dataset
                                parfor b = 1:B
                                    % (a) Resample based on TP2 
                                    idx = randi(N, [N,1]);
                                    bootData_tp2 = simulwith_TP2(idx,:);
                                    bootData_dp2 = simulwith_DP2(idx,:);

                                    % (b) Build contingency table from bootstrap sample
                                    [matching_b_tp2, ~, ~] = crosstab(bootData_tp2.edu, bootData_tp2.edu_sp);
                                    % (c) Compute the log-likelihoods on the bootstrap table
                                    [lr_uncon_b_tp2, lr_tp2_b, ~,~,~] = L_siow(matching_b_tp2, param,2);

                                    [matching_b_dp2, ~, ~] = crosstab(bootData_dp2.edu, bootData_dp2.edu_sp);
                                    [lr_uncon_b_dp2, ~, lr_dp2_b,~,~] = L_siow(matching_b_dp2, param,3);

                                    % (d) Form the LR test statistics
                                    LR_TP2_dis(b,r,a,k) = 2 * (lr_uncon_b_tp2 - lr_tp2_b);
                                    LR_DP2_dis(b,r,a,k) = 2 * (lr_uncon_b_dp2 - lr_dp2_b);
                                end
            % Verdict
            %     pval_TP2(r,a,k)=pvalgen(LR_TP2(r,a,k),LR_TP2_dis(:,r,a,k));
            %     pval_DP2(r,a,k)=pvalgen(LR_DP2(r,a,k),LR_DP2_dis(:,r,a,k));
            % % Given distribution, one can find a simulated p-value for a
            % given statistics (Interpolation?)
            
        end
    end
end


%%

for a = 1:sz_alpha
    alpha_prime = alpha_vec(a);
    for k = 1:sz_n
        for r = 1:R

         if LR_DP2(r,a,k) <0
LR_DP2(r,a,k)=0;
         end
         pval_TP2(r,a,k)=pvalgen(LR_TP2(r,a,k),LR_TP2_dis(:,r,a,k));
         pval_DP2(r,a,k)=pvalgen(LR_DP2(r,a,k),LR_DP2_dis(:,r,a,k));
        end

        average_TP2(a,k)=mean(LR_TP2(1:100,a,k),'omitnan');
        average_DP2(a,k)=mean(LR_DP2(1:100,a,k));
        average_pval_TP2(a,k)=mean(pval_TP2(1:100,a,k));
        average_pval_DP2(a,k)=mean(pval_DP2(1:100,a,k));

    end
end





%% SECTION 6: Stern & Kang test using Chi-square
T1s1_null = NaN(M,F,R,sz_n);

for a = 1:sz_alpha
    for k = 1:sz_n
        for r = 1:R
            % We use s_data(:,:,r,a,k) as the empirical distribution
            T1s1_stat(r,a,k) = tstat( contingency_full(:,:,r,a,k)/sum(n_vec(k),"all"),1,n_vec(k),M);
            p_values(r,a,k)  = 1 - chi2cdf(T1s1_stat(r,a,k), (M-1)*(F-1));
        end
        average_pvalues(a,k) = mean(p_values(1:100,a,k));
        average_chi(a,k)     = mean(T1s1_stat(1:100,a,k));
    end
end


%% SECTION 6: Stern & Kang test using LR 
T1s1_null = NaN(M,F,R,sz_n);

for a = 1:sz_alpha
    for k = 1:sz_n
        for r = 1:R
   % ADDED CODE: Now do Stern & Kang "LR" test (unrestricted vs random).
            % (1) Unrestricted p^u: c_{ij} / N
            % (2) Random p^r: row_i * col_j / N^2
            contingency_r=contingency_full(:,:,r,a,k);
            cSum = sum(contingency_r(:));
            pU = contingency_r / cSum;  % MxF
            rowSums = sum(contingency_r,2); % Mx1
            colSums = sum(contingency_r,1); % 1xF
            pR = zeros(M,F);
            for mm=1:M
                for ff=1:F
                    pR(mm,ff) = (rowSums(mm)/cSum)*(colSums(ff)/cSum);
                end
            end
            % LR = 2 sum c_{ij} log( p^u_{ij} / p^r_{ij} )
            logRatio = 0; 
            for mm=1:M
                for ff=1:F
                    if pU(mm,ff)>0 && pR(mm,ff)>0
                        logRatio = logRatio + contingency_r(mm,ff)*log(pU(mm,ff)/pR(mm,ff));
                    end
                end
            end
            LR_SK(r,a,k) = 2*logRatio;

            % p-value from chi2 with df=(M-1)(F-1)
            dof = M*F-1;
            pVal_SK(r,a,k) = 1 - chi2cdf(LR_SK(r,a,k), dof);
        end

    end
end
%% Now we can compute the average LR_SK and p-value, just like the others
average_LR_SK  = NaN(sz_alpha,sz_n);
average_pValSK = NaN(sz_alpha,sz_n);

for a = 1:sz_alpha
    for k = 1:sz_n
        average_LR_SK(a,k)  = mean(LR_SK(:,a,k),"omitnan");
        average_pValSK(a,k) = mean(pVal_SK(:,a,k),"omitnan");
    end
end
%%

createLatexTableWithStars(alpha_vec,n_vec,average_chi',average_pvalues')

createLatexTableWithStars(alpha_vec,n_vec,average_TP2',average_pval_TP2')

createLatexTableWithStars(alpha_vec,n_vec,average_DP2',average_pval_DP2')

createLatexTableWithStars(alpha_vec, n_vec, average_LR_SK', average_pValSK')

%%
% %% SECTION 8: Siow (2015) approach (using CVX)
% siow_LR_result = NaN(R,sz_alpha,sz_n);
% 
% for a = 1:sz_alpha
%     for l = 1:sz_n
%         parfor r = 1:R
%             c = contingency_full(:,:,r,a,l);
%         [lr_uncon, lr_tp2, lr_dp2] = L_siow(c, param,1);
%            LR_result_tp2(r,a,l) =2 * (lr_uncon - lr_tp2);
%            LR_result_dp2(r,a,l) =2 * (lr_uncon - lr_dp2);
%         end
%     end
% end
% 
% for a = 1:sz_alpha
%     for k = 1:sz_n
%         average_LR(a,k) = mean(siow_LR_result(:,a,k));
%     end
% end
% 
% createPValueTable(alpha_vec, n_vec, average_LR', true)

% %% SECTION 9: Create tables with star significance
% critVals = zeros(sz_alpha,sz_n,3);
critVals(:,:,1) = 0.05;   % '*'
critVals(:,:,2) = 0.01;   % '**'
critVals(:,:,3) = 0.001;  % '***'
% 
% createPValueTable(alpha_vec, n_vec, average_pvalues', true)
createPValueTableWithStars(alpha_vec, n_vec, average_pvalues, critical_chi, true);
% % createPValueTableWithStars(alpha_vec, n_vec, average_chi,       critical_chi, true);
% createPValueTableWithStars(alpha_vec, n_vec, average_LR,        critical_TP2, true);
% createPValueTableWithStars(alpha_vec, n_vec, average_LR,        critical_DP2, true);
% createPValueTableWithStars(alpha_vec, n_vec, average_chi,       critical_chi, true);

%%
% save("section3_results")

%{ OBSOLETE 

% %% SECTION 6: Pre-compute the Chi-bar square for the model 
% sz_sig        = 3;
% weights_TP2   = NaN((M-1)*(F-1)+1+1,sz_alpha,sz_n,sz_sig);
% critical_TP2  = NaN(sz_alpha,sz_n,sz_sig);
% weights_DP2   = NaN((M-1)+1,sz_alpha,sz_n,sz_sig);
% critical_DP2  = NaN(sz_alpha,sz_n,sz_sig);
% critical_chi  = NaN(sz_alpha,sz_n,sz_sig);
% c_table       = contingency_full;
% sigvec        = [0.1, 0.05, 0.01];
% 
% for ab = 1:sz_alpha
%     for cd = 1:sz_n
%         for ef = 1:sz_sig
%             [weights_TP2(:,ab,cd,ef), critical_TP2(ab,cd,ef)] = ...
%                 chibar_analysis(c_table(:,:,:,ab,cd), sigvec(ef),'critical_value','TP2');
%         end
%     end
% end
% 
% for ab = 1:sz_alpha
%     for cd = 1:sz_n
%         for ef = 1:sz_sig
%             [weights_DP2(:,ab,cd,ef), critical_DP2(ab,cd,ef)] = ...
%                 chibar_analysis(c_table(:,:,:,ab,cd), sigvec(ef),'critical_value','DP2');
%         end
%     end
% end
% 
% for ab = 1:sz_alpha
%     for cd = 1:sz_n
%         for ef = 1:sz_sig
%             critical_chi(ab,cd,ef) = chi2inv(1 - sigvec(ef),(M-1)*(F-1));
%         end
%     end
% end
