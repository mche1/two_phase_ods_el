function ee=eelog(pop,beta,alpha)
N=length(pop);
lb=length(beta);
% beta=param(1:lb);
% alpha=param((lb+1):end);
la=length(alpha);
y=pop(:,1);
o=ones(N,1);
z=pop(:,2);
x1=pop(:,3);
%x2=pop(:,4);
s1=pop(:,5);
s2=pop(:,6);
R=pop(:,end);
X=[o,z,x1];
X=X(:,1:lb);
Z=[o,y,s1,s2,y.*s1,y.*s2];
Z1=[o,o,s1,s2,s1,s2];
Z0=[o,zeros(N,1),s1,s2,zeros(N,2)];
Z=Z(:,1:la);
Z1=Z1(:,1:la);
Z0=Z0(:,1:la);
Sb=X.*repmat(R.*(y-expit(woffset(Z1,Z0,alpha)+X*beta)),[1 lb]);
wa=jacobianestim(@(x)woffset(Z1,Z0,x),alpha);
Sa1=wa.*repmat(R.*(y-expit(woffset(Z1,Z0,alpha)+X*beta)),[1 la]);
Sa=Z.*repmat((R-expit(Z*alpha)),[1 la]);
ee=[Sb,Sa,Sa1];
function w=woffset(Z1,Z0,alpha)
w=log(expit(Z1*alpha)./expit(Z0*alpha));
