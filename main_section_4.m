%% SECTION 4: Same-sex marriage — bootstrap SK + Siow TP2/DP2
clc; clear;
cvx_setup;

%% 1) Load data and build contingency tables
maindata   = readtable('../data/samesex.csv');
data_table = maindata(:, {'edu','edu_sp','sex'});

M = 5; F = 5;

% pooled
[matching_main, ~, ~] = crosstab(data_table.edu, data_table.edu_sp);

% male-only
ma_data = data_table(strcmp(data_table.sex,'male'), :);
[matching_male, ~, ~] = crosstab(ma_data.edu, ma_data.edu_sp);

% female-only
fe_data = data_table(strcmp(data_table.sex,'female'), :);
[matching_female, ~, ~] = crosstab(fe_data.edu, fe_data.edu_sp);

%% 2) Stern–Kang tests (pseudo-Wald and LR) with parametric bootstrap
%    Bootstrap under H0: independence using plug-in product of margins
B_SK = 1000; rng(123);

SK_both  = bootstrap_SK(matching_main,   M, F, B_SK, 123);
SK_male  = bootstrap_SK(matching_male,   M, F, B_SK, 456);
SK_fem   = bootstrap_SK(matching_female, M, F, B_SK, 789);

SternKang = struct();
SternKang.both.T = SK_both.wald_T;  SternKang.both.p = SK_both.wald_p;
SternKang.male.T = SK_male.wald_T;  SternKang.male.p = SK_male.wald_p;
SternKang.female.T = SK_fem.wald_T; SternKang.female.p = SK_fem.wald_p;

SternKangLR = struct();
SternKangLR.both.T = SK_both.lr_T;  SternKangLR.both.p = SK_both.lr_p;
SternKangLR.male.T = SK_male.lr_T;  SternKangLR.male.p = SK_male.lr_p;
SternKangLR.female.T = SK_fem.lr_T; SternKangLR.female.p = SK_fem.lr_p;

fprintf('\n[Both] SK Wald 95%%=%.2f,99%%=%.2f; SK LR 95%%=%.2f,99%%=%.2f\n', ...
    SK_both.wald_crit95, SK_both.wald_crit99, SK_both.lr_crit95, SK_both.lr_crit99);

%% 3) Siow (2015) LR tests with parametric bootstrap 
param.M = M; param.F = F;
parpool(80);
B = 1000; rng(123);

results = struct([]);
samples = {'both','male','female'};

for i = 1:numel(samples)
    switch samples{i}
        case 'both'
            c_data = data_table;      c_mat = matching_main;   tag = 'Both Sample';
        case 'male'
            c_data = ma_data;         c_mat = matching_male;   tag = 'Male Sample';
        case 'female'
            c_data = fe_data;         c_mat = matching_female; tag = 'Female Sample';
    end

    % unrestricted vs restricted fits (TP2 / DP2)
    [lr_uncon, lr_tp2, lr_dp2, mu_con, mu_dpcon] = L_siow(c_mat, param, 1);
    LR_result_tp2 = 2*(lr_uncon - lr_tp2);
    LR_result_dp2 = 2*(lr_uncon - lr_dp2);

    % parametric bootstrap under TP2 / DP2 worlds
    N = size(c_data,1);
    LR_TP2_dis = zeros(B,1);
    LR_DP2_dis = zeros(B,1);

    parfor b = 1:B
        % TP2 world
        boot_tp2 = generateTableWithEdu(mu_con, N);
        match_b_tp2 = crosstab(boot_tp2.edu, boot_tp2.edu_sp);
        [lu_b_tp2, lp_b_tp2, ~, ~, ~] = L_siow(match_b_tp2, param, 2);
        LR_TP2_dis(b) = 2*(lu_b_tp2 - lp_b_tp2);

        % DP2 world
        boot_dp2 = generateTableWithEdu(mu_dpcon, N);
        match_b_dp2 = crosstab(boot_dp2.edu, boot_dp2.edu_sp);
        [lu_b_dp2, ~, lp_b_dp2, ~, ~] = L_siow(match_b_dp2, param, 3);
        LR_DP2_dis(b) = 2*(lu_b_dp2 - lp_b_dp2);
    end

    % empirical p-values and CVs
    pval_TP2 = (1 + sum(LR_TP2_dis >= LR_result_tp2)) / (B + 1);
    pval_DP2 = (1 + sum(LR_DP2_dis >= LR_result_dp2)) / (B + 1);

    results(i).sample = tag;
    results(i).LR_result_tp2 = LR_result_tp2;
    results(i).LR_result_dp2 = LR_result_dp2;
    results(i).pval_TP2 = pval_TP2;
    results(i).pval_DP2 = pval_DP2;
    results(i).CI_TP2_95 = prctile(LR_TP2_dis,95);
    results(i).CI_TP2_99 = prctile(LR_TP2_dis,99);
    results(i).CI_DP2_95 = prctile(LR_DP2_dis,95);
    results(i).CI_DP2_99 = prctile(LR_DP2_dis,99);

    fprintf('\n=== %s ===\n', tag);
    fprintf('LR(TP2)=%.3f  [p_boot=%.3f]\n', LR_result_tp2, pval_TP2);
    fprintf('LR(DP2)=%.3f  [p_boot=%.3f]\n', LR_result_dp2, pval_DP2);
end

SiowResults = results; %#ok<NASGU>
save("results_section4","results","SternKang","SternKangLR");

%% 4) LaTeX table (uses bootstrap p-values for all tests)
sampleNames     = {"Both","Male","Female"};
sternKangFields = {"both","male","female"};
fid = fopen("Tab_SameSexMarriage.tex",'w');

fprintf(fid,'\\begin{table}[!htbp]\\centering\n');
fprintf(fid,'\\caption{Comparison of Stern--Kang and Siow Tests by Sample Type}\n');
fprintf(fid,'\\begin{tabular}{lcccc}\n');
fprintf(fid,'\\hline\\hline\n');
fprintf(fid,'Sample & Test & Statistic & $p$-value & Verdict \\\\ \\hline\n');

for i = 1:numel(sampleNames)
    f = sternKangFields{i};

    % SK pseudo-Wald (bootstrap)
    fprintf(fid,'%s & Kang--Stern (pseudo--Wald) & %s & %s & P \\\\\n', ...
        sampleNames{i}, formatVal(SternKang.(f).T), formatVal(SternKang.(f).p));

    % SK LR (bootstrap)
    fprintf(fid,'%s & Kang--Stern (LR) & %s & %s & P \\\\\n', ...
        sampleNames{i}, formatVal(SternKangLR.(f).T), formatVal(SternKangLR.(f).p));

    % Siow TP2 (bootstrap)
    fprintf(fid,'%s & Siow TP2 & %s & %s & R \\\\\n', ...
        sampleNames{i}, formatVal(results(i).LR_result_tp2), formatVal(results(i).pval_TP2));

    % Siow DP2 (bootstrap)
    fprintf(fid,'%s & Siow DP2 & %s & %s & R \\\\\n', ...
        sampleNames{i}, formatVal(results(i).LR_result_dp2), formatVal(results(i).pval_DP2));

    if i < numel(sampleNames), fprintf(fid,'\\hline\n'); end
end

fprintf(fid,'\\hline\\hline\n');
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\label{tab:stern_kang_siow_all}\n');
fprintf(fid,'\\end{table}\n');
fclose(fid);
