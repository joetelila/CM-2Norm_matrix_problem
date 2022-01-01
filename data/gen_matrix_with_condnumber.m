%{
Authors: Dawit Anelay
         Marco Pitex
         Yohannis Telila
%}

m1 = 2*rand(1000,1000)-1 + 2*rand(1000,1000)-1;     
m1 = m1+m1';                     
cond_number = 2e15;              % desired condition number
[u s v] = svd(m1);
s = diag(s);           % s is vector
% ===== linear stretch of existing s
s = s(1)*( 1-((cond_number-1)/cond_number)*(s(1)-s)/(s(1)-s(end))) ;
% =====
s = diag(s);           % back to matrix
m1_illcond = u*s*v';
cond(m1_illcond)
dlmwrite('M1.txt', m1_illcond, 'delimiter', '\t', 'precision', 16)
disp("[success] M1 generated and saved.")