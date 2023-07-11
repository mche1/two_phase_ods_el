% function s=su(pop,para,alpha,c,theta)
%  u=u(pop,para,alpha,c,theta)
%  s=sum(u);
% function [sume]=scmle(pop,param,alpha,c)
% s=size(pop);
% [e]=scondlee(pop,param,alpha,c);
% lp=length(param);
% sume=sum(e(:,1:lp));
% %sume1(4:6)=sum(e(:,7:9))-sum(e(:,4:6));

function [sume]=scml0(pop,beta,alpha)
s=size(pop);
[e]=eelog(pop,beta,alpha);
lb=length(beta);
la=length(alpha);
sume=sum(e(:,1:lb));