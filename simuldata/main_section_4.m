%% SECTION 1: Load data
clc
clear
cvx_setup
maindata = readtable('../data/samesex.csv');
var_names = maindata.Properties.VariableNames;
data_table = maindata(:, {'edu', 'edu_sp','sex'});

ma_data = data_table(strcmp(data_table.sex,'male'), :);
fe_data = data_table(strcmp(data_table.sex,'female'), :);

% Both
[matching_main, ~, ~] = crosstab(data_table.edu, data_table.edu_sp);
main_data = matching_main / sum(matching_main,"all");
main_data(main_data==0)=eps;

% Male
[matching_male, ~, ~] = crosstab(ma_data.edu, ma_data.edu_sp);
male_data = matching_male / sum(matching_male,"all");
male_data(male_data==0)=eps;

% Female
[matching_female, ~, ~] = crosstab(fe_data.edu, fe_data.edu_sp);
female_data = matching_female / sum(matching_female,"all");
female_data(female_data==0)=eps;

% Stern and Kang Test using Chi-Square
M = 5; 
F = 5;

% (1) Both
SternKang.both.T = tstat(main_data,1,sum(matching_main,"all"),M);
SternKang.both.p = 1 - chi2cdf(SternKang.both.T, (M-1)*(F-1));

% (2) Male
SternKang.male.T = tstat(male_data,1,sum(matching_male,"all"),M);
SternKang.male.p = 1 - chi2cdf(SternKang.male.T, (M-1)*(F-1));

% (3) Female
SternKang.female.T = tstat(female_data,1,sum(matching_female,"all"),M);
SternKang.female.p = 1 - chi2cdf(SternKang.female.T, (M-1)*(F-1));

% Display Stern & Kang results
disp('=== Stern & Kang Chi-Square Test ===')
disp(SternKang.male)
disp(SternKang.female)
disp(SternKang.both)



%% ADD: Stern & Kang LR test (random vs. unconstrained)
SternKangLR = struct();

% Helper function to do the random vs. unconstrained LR:
% cMat is integer counts, MxF
% Returns [LR_value, p_value].
doSKLR = @(cMat) dealSKLR(cMat);

% 1) Both
[SKLR_val, SKLR_pval] = doSKLR(matching_main);
SternKangLR.both.T = SKLR_val;
SternKangLR.both.p = SKLR_pval;

% 2) Male
[SKLR_val, SKLR_pval] = doSKLR(matching_male);
SternKangLR.male.T = SKLR_val;
SternKangLR.male.p = SKLR_pval;

% 3) Female
[SKLR_val, SKLR_pval] = doSKLR(matching_female);
SternKangLR.female.T = SKLR_val;
SternKangLR.female.p = SKLR_pval;

disp('=== Stern & Kang LR Test (Random vs Unrestricted) ===')
disp(SternKangLR.male)
disp(SternKangLR.female)
disp(SternKangLR.both)

%% Siow (2015) LR Test + Bootstrap for each sample type
sampleTypes = {"both", "male", "female"};
%% Siow (2015) LR Test + Bootstrap for each sample type
%  We create a small loop to handle: 'both', 'male', 'female'

sampleTypes = {"both", "male", "female"};

param.M = 5;
param.F = 5;
parpool(80)
B = 1000;           % number of bootstrap replications (example)
rng(123);         % set seed for reproducibility
% Pre-allocate structure to store results clearly
results = struct();

for i = 1:numel(sampleTypes)
    
    % 1. Select sample
    switch sampleTypes{i}
        case "both"
            c_data    = data_table;
            c_mat     = matching_main;
            sampleTag = 'Both Sample';
        case "male"
            c_data    = ma_data;
            c_mat     = matching_male;
            sampleTag = 'Male Sample';
        case "female"
            c_data    = fe_data;
            c_mat     = matching_female;
            sampleTag = 'Female Sample';
    end
    
    % 2. Compute original LR stats (unrestricted, TP2, DP2)
    [lr_uncon, lr_tp2, lr_dp2, mu_con, mu_dpcon] = L_siow(c_mat, param, 1);
    LR_result_tp2 = 2 * (lr_uncon - lr_tp2);
    LR_result_dp2 = 2 * (lr_uncon - lr_dp2);
    
    % 3. Set up bootstrap
    N = size(c_data,1);
    LR_TP2_dis = zeros(B,1);
    LR_DP2_dis = zeros(B,1);

    parfor b = 1:B
        
        % (a) Generate parametric sample from mu_con (TP2)
        bootData_tp2 = generateTableWithEdu(mu_con, N);
        matching_b_tp2 = crosstab(bootData_tp2.edu, bootData_tp2.edu_sp);
        [lr_uncon_b_tp2, lr_tp2_b, ~, ~, ~] = L_siow(matching_b_tp2, param, 2);
        LR_TP2_dis(b) = 2 * (lr_uncon_b_tp2 - lr_tp2_b);
        
        % (b) Generate parametric sample from mu_dpcon (DP2)
        bootData_dp2 = generateTableWithEdu(mu_dpcon, N);
        matching_b_dp2 = crosstab(bootData_dp2.edu, bootData_dp2.edu_sp);
        [lr_uncon_b_dp2, ~, lr_dp2_b, ~, ~] = L_siow(matching_b_dp2, param, 3);
        LR_DP2_dis(b) = 2 * (lr_uncon_b_dp2 - lr_dp2_b);
    end
    
    % 4. Bootstrap statistics summaries
    CI_TP2_95  = prctile(LR_TP2_dis, 95);
    CI_TP2_99  = prctile(LR_TP2_dis, 99);
    pval_TP2   = pvalgen(LR_result_tp2, LR_TP2_dis);
    
    CI_DP2_95  = prctile(LR_DP2_dis, 95);
    CI_DP2_99  = prctile(LR_DP2_dis, 99);
    pval_DP2   = pvalgen(LR_result_dp2, LR_DP2_dis);
    
    % 5. Store the results clearly in the structure
    results(i).sample    = sampleTag;
    results(i).LR_TP2_dis = LR_TP2_dis;
    results(i).LR_DP2_dis = LR_DP2_dis;
    results(i).LR_result_tp2 = LR_result_tp2;
    results(i).LR_result_dp2 = LR_result_dp2;
    results(i).CI_TP2_95 = CI_TP2_95;
    results(i).CI_TP2_99 = CI_TP2_99;
    results(i).pval_TP2  = pval_TP2;
    results(i).CI_DP2_95 = CI_DP2_95;
    results(i).CI_DP2_99 = CI_DP2_99;
    results(i).pval_DP2  = pval_DP2;

    % 6. Print Results
    fprintf('\n=== %s ===\n', sampleTag);
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
    
    save("results_section4")

    SiowResults=results
%% Create final LaTeX table

sampleTypes      = {"Both","Male","Female"};
sternKangFields  = {"both","male","female"};

fid = fopen("Tab_SameSexMarriage.tex",'w');

fprintf(fid,'\\begin{table}[!htbp]\\centering\n');
fprintf(fid,'\\caption{Comparison of Stern & Kang vs. Siow Tests by Sample Type}\n');
fprintf(fid,'\\begin{tabular}{lccccc}\n');
fprintf(fid,'\\hline\\hline\n');
fprintf(fid,'Sample & Test & Statistic & $p$-value & Assortative Mating & Comments \\\\ \\hline\n');

for i = 1:numel(sampleTypes)
    field = sternKangFields{i};

    % 1) Stern & Kang pseudo-Wald
    fprintf(fid,'%s & Kang--Stern (Wald) & %s & %s & A & pseudo-Wald \\\\\n', ...
        sampleTypes{i}, ...
        formatVal(SternKang.(field).T), ...
        formatVal(SternKang.(field).p));

    % 2) Stern & Kang LR
    fprintf(fid,'%s & Kang--Stern (LR)   & %s & %s & A & random vs unconstrained \\\\\n', ...
        sampleTypes{i}, ...
        formatVal(SternKangLR.(field).T), ...
        formatVal(SternKangLR.(field).p));

    % 3) Siow LR TP2
    fprintf(fid,'%s & Siow LR (TP2) & %s & %s & R & supermod. test \\\\\n', ...
        sampleTypes{i}, ...
        formatVal(SiowResults(i).LR_result_tp2), ...
        formatVal(SiowResults(i).pval_TP2));

    % 4) Siow LR DP2
    fprintf(fid,'%s & Siow LR (DP2) & %s & %s & R & supermod. test \\\\\n', ...
        sampleTypes{i}, ...
        formatVal(SiowResults(i).LR_result_dp2), ...
        formatVal(SiowResults(i).pval_DP2));

    if i < numel(sampleTypes)
        fprintf(fid,'\\hline\n');
    end
end

fprintf(fid,'\\hline\\hline\n');
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\label{tab:stern_kang_siow_all}\n');
fprintf(fid,'\\end{table}\n');

fclose(fid);
    %%


    sampleTypes = {"Both", "Male", "Female"};

sampleTypes = {"Both", "Male", "Female"};
sternKangFields = {"both", "male", "female"};
fid = fopen("Tab_SameSexMarriage.tex", 'w');


fprintf(fid,'\\begin{table}[!htbp]\\centering\n');
fprintf(fid,'\\caption{Comparison of Stern & Kang vs. Siow Tests by Sample Type}\n');
fprintf(fid,'\\begin{tabular}{lcccc}\n');
fprintf(fid,'\\hline\\hline\n');
fprintf(fid,'Sample & Test & Statistic & p-value & Assortative Mating \\\\ \\hline\n');

for i = 1:numel(sampleTypes)
    % Stern & Kang
    field = sternKangFields{i};
    fprintf(fid,'%s & Stern & Kang & %.3f & %.3f & A \\\\ \n', sampleTypes{i}, SternKang.(field).T, SternKang.(field).p);

    % Siow LR Test TP2
    fprintf(fid,'%s & Siow LR TP2 & %.3f & %.3f & R \\\\ \n', sampleTypes{i}, SiowResults(i).LR_result_tp2, SiowResults(i).pval_TP2);

    % Siow LR Test DP2
    fprintf(fid,'%s & Siow LR DP2 & %.3f & %.3f & R \\\\ \n', sampleTypes{i}, SiowResults(i).LR_result_dp2, SiowResults(i).pval_DP2);
  if i < numel(sampleTypes)
        fprintf(fid,'\\hline\n');
  end
  
end

fprintf(fid,'\\hline\\hline\n');
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\label{tab:stern_kang_siow_all}\n');
fprintf(fid,'\\end{table}\n');

fclose(fid);


end
