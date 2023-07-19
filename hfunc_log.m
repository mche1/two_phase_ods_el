 function h=hfunc_log(y,x,theta)
 sp=length(x);
 o=ones(sp(1),1);
 cov=[o,x];
 muy=expit(cov*theta);
 h=repmat(y-muy,[1,2]).*cov;
 