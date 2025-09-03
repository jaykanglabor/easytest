%% SECTION 1: Load data
clc
clear
cvx_setup
maindata = readtable('../data/cps_male_2020_incrank.csv');
var_names = maindata.Properties.VariableNames;
data_table = maindata(:, {'own_wage_decile', 'own_wage_quintile','spouse_wage_decile','spouse_wage_quintile','educ_5', 'educ_sp_5'});

[matching_main, ~, ~] = crosstab(data_table.own_wage_quintile, data_table.spouse_wage_quintile);
main_data = matching_main / sum(matching_main,"all");
main_data(main_data==0)=eps;


[matching_main_edu, ~, ~] = crosstab(data_table.educ_5, data_table.educ_sp_5);
main_data_edu = matching_main_edu / sum(matching_main_edu,"all");
main_data_edu(main_data_edu==0)=eps;

cell_inc=main_data/sum(main_data,"all");
cell_edu=main_data_edu/sum(main_data_edu,"all");
%% Stern and Kang Test using Chi-Square
M = 5; 
F = 5;


SternKang.both.T = tstat(main_data,1,sum(matching_main,"all"),M);
SternKang.both.p = 1 - chi2cdf(SternKang.both.T, (M-1)*(F-1));


% Display Stern & Kang results
disp('=== Stern & Kang Chi-Square Test ===')
disp(SternKang.both)

% ADD: Stern & Kang LR Test (Random vs. Unrestricted)
[SKLR_val, SKLR_pval] = doSKLR(matching_main);
SK_LR.T = SKLR_val;
SK_LR.p = SKLR_pval;

disp('=== Stern & Kang LR Test (Random vs Unrestricted) ===')
disp(SK_LR)



%% Siow (2015) LR Test + Bootstrap for each sample type
%  We create a small loop to handle: 'both', 'male', 'female'


param.M = 5;
param.F = 5;
% parpool(80)
B = 10;           % number of bootstrap replications (example)
rng(123);         % set seed for reproducibility
% Pre-allocate structure to store results clearly
results = struct();

    % 2. Compute original LR stats (unrestricted, TP2, DP2)
    [lr_uncon, lr_tp2, lr_dp2, mu_con, mu_dpcon] = L_siow(matching_main, param, 1);
    LR_result_tp2 = 2 * (lr_uncon - lr_tp2);
    LR_result_dp2 = 2 * (lr_uncon - lr_dp2);
    
    % 3. Set up bootstrap
    N = sum(matching_main,"all");
    LR_TP2_dis = zeros(B,1);
    LR_DP2_dis = zeros(B,1);
    parfor b = 1:B
        
        % (a) Generate parametric sample from mu_con (TP2)
        bootData_tp2 = generateTableWithWage(mu_con, N);
        matching_b_tp2 = crosstab(bootData_tp2.own_wage, bootData_tp2.spouse_wage);
        [lr_uncon_b_tp2, lr_tp2_b, ~, ~, ~] = L_siow(matching_b_tp2, param, 2);
        LR_TP2_dis(b) = 2 * (lr_uncon_b_tp2 - lr_tp2_b);
        
        % (b) Generate parametric sample from mu_dpcon (DP2)
        bootData_dp2 = generateTableWithWage(mu_dpcon, N);
        matching_b_dp2 = crosstab(bootData_dp2.own_wage, bootData_dp2.spouse_wage);
        [lr_uncon_b_dp2, ~, lr_dp2_b, ~, ~] = L_siow(matching_b_dp2, param, 3);
        LR_DP2_dis(b) = 2 * (lr_uncon_b_dp2 - lr_dp2_b);
    end
    
       for b = 1:B
        
            if LR_DP2_dis(b)<0
                LR_DP2_dis(b)=0;
            end
            if LR_TP2_dis(b)<0
                LR_TP2_dis(b)=0;
            end

         end

    % 4. Bootstrap statistics summaries
    CI_TP2_95  = prctile(LR_TP2_dis, 95);
    CI_TP2_99  = prctile(LR_TP2_dis, 99);
    pval_TP2   = pvalgen2(LR_result_tp2, LR_TP2_dis);
    
    CI_DP2_95  = prctile(LR_DP2_dis, 95);
    CI_DP2_99  = prctile(LR_DP2_dis, 99);
    pval_DP2   = pvalgen2(LR_result_dp2, LR_DP2_dis);
    
    % 5. Store the results clearly in the structure
    results.LR_TP2_dis = LR_TP2_dis;
    results.LR_DP2_dis = LR_DP2_dis;
    results.LR_result_tp2 = LR_result_tp2;
    results.LR_result_dp2 = LR_result_dp2;
    results.CI_TP2_95 = CI_TP2_95;
    results.CI_TP2_99 = CI_TP2_99;
    results.pval_TP2  = pval_TP2;
    results.CI_DP2_95 = CI_DP2_95;
    results.CI_DP2_99 = CI_DP2_99;
    results.pval_DP2  = pval_DP2;

    % 6. Print Results
    fprintf('Original data:\n');
    fprintf('  LR(TP2) = %.3f\n', LR_result_tp2);
    fprintf('  LR(DP2) = %.3f\n\n', LR_result_dp2);
    
    fprintf('TP2 Model Bootstrap Results:\n');
    fprintf('  95%% critical value = %.3f\n', CI_TP2_95);
    fprintf('  99%% critical value = %.3f\n', CI_TP2_99);
    fprintf('  p-value (empirical) = %.3f\n\n', pval_TP2);
    
    fprintf('DP2 Model Bootstrap Results:\n');
    fprintf('  95%% critical value = %.3f\n', CI_DP2_95);
    fprintf('  99%% critical value = %.3f\n', CI_DP2_99);
    fprintf('  p-value (empirical) = %.3f\n\n', pval_DP2);
    

    % Writing LaTeX Table
fid = fopen("Tab_IncRank.tex", 'w');
% Writing LaTeX Table

fprintf(fid,'\\begin{table}[!htbp]\\centering\n');
fprintf(fid,'\\caption{Comparison of Stern & Kang vs. Siow Tests}\n');
fprintf(fid,'\\begin{tabular}{lccc}\n');
fprintf(fid,'\\hline\\hline\n');
fprintf(fid,'Test & Statistic & p-value & Assortative Mating \\\\ \\hline\n');

fprintf(fid,'Stern & Kang (Chi-square) & %.3f & %.3f & A \\\\ \n', SternKang.both.T, SternKang.both.p);

fprintf(fid,'Siow LR TP2 & %.3f & %.3f & R \\\\ \n', LR_result_tp2, pval_TP2);
fprintf(fid,'Siow LR DP2 & %.3f & %.3f & R \\\\ \n', LR_result_dp2, pval_DP2);

fprintf(fid,'\\hline\\hline\n');
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\label{tab:stern_kang_siow}\n');
fprintf(fid,'\\end{table}\n');

fclose(fid);


    % save("results_section4b")

    %%


fprintf('Original data:\n');
fprintf('  LR(TP2) = %.3f\n', LR_result_tp2);
fprintf('  LR(DP2) = %.3f\n\n', LR_result_dp2);

fprintf('TP2 Model Bootstrap Results:\n');
fprintf('  95%% critical value = %.3f\n', CI_TP2_95);
fprintf('  99%% critical value = %.3f\n', CI_TP2_99);
fprintf('  p-value (empirical) = %.3f\n\n', pval_TP2);

fprintf('DP2 Model Bootstrap Results:\n');
fprintf('  95%% critical value = %.3f\n', CI_DP2_95);
fprintf('  99%% critical value = %.3f\n', CI_DP2_99);
fprintf('  p-value (empirical) = %.3f\n\n', pval_DP2);

fid = fopen("Tab_IncRank.tex",'w');

fprintf(fid,'\\begin{table}[!htbp]\\centering\n');
fprintf(fid,'\\caption{Comparison of Stern & Kang vs. Siow Tests (Income Ranks)}\n');
fprintf(fid,'\\begin{tabular}{lccc}\n');
fprintf(fid,'\\hline\\hline\n');
fprintf(fid,'Test & Statistic & p-value & Assortative Mating \\\\ \\hline\n');

% 1) Stern & Kang (Chi-Square)
fprintf(fid,'Kang--Stern (pseudo--Wald) & %s & %s & A \\\\ \n', ...
        formatVal(SternKang.both.T), formatVal(SternKang.both.p));

% 2) NEW: Stern & Kang (LR)
fprintf(fid,'Kang--Stern (LR) & %s & %s & A \\\\ \n', ...
        formatVal(SK_LR.T), formatVal(SK_LR.p));

% 3) Siow LR TP2
fprintf(fid,'Siow TP2 & %s & %s & R \\\\ \n', ...
        formatVal(results.LR_result_tp2), formatVal(results.pval_TP2));

% 4) Siow LR DP2
fprintf(fid,'Siow DP2 & %s & %s & R \\\\ \n', ...
        formatVal(results.LR_result_dp2), formatVal(results.pval_DP2));

fprintf(fid,'\\hline\\hline\n');
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\label{tab:stern_kang_siow}\n');
fprintf(fid,'\\end{table}\n');

fclose(fid);

%% Subfunction for Kang--Stern LR Test
function [lrVal, pVal] = doSKLR(cMat)
    % cMat: integer counts (M=10,F=10)
    sumC = sum(cMat(:));
    pU   = cMat / sumC;  % Unrestricted MLE
    rowS = sum(cMat,2);
    colS = sum(cMat,1);
    [M,F] = size(cMat);
    pR = zeros(M,F);
    for i=1:M
        for j=1:F
            pR(i,j) = (rowS(i)/sumC)*(colS(j)/sumC);
        end
    end
    logSum = 0;
    for i=1:M
        for j=1:F
            if pU(i,j)>0 && pR(i,j)>0
                logSum = logSum + cMat(i,j)*log( pU(i,j)/pR(i,j) );
            end
        end
    end
    lrVal = 2*logSum;
    df    = (M-1)*(F-1);
    pVal  = 1 - chi2cdf(lrVal, df);
end

