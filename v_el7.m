function v2=v_el7(pop,beta,alpha,theta) 
lb=length(beta);
 la=length(alpha);
        uf=ufunc_log(pop,beta,alpha,theta);
        s=eelog(pop,beta,alpha);
        sb=s(:,1:lb);
        sa1=s(:,(lb+1):(lb+la));
        sa=s(:,(lb+la+1):end);
        h=hfunc_log(pop(:,1),pop(:,3),theta);
        
        u=[uf,sb,sa-sa1,h];
          siz=size(u);
          sizu=size(uf);
          sizs=size(s);
          sizh=size(h);

            J11=jacobianestim(@(x)sum(ufunc_log(pop,x,alpha,theta)),beta);
            J21=jacobianestim(@(x)sum(eelog(pop,x,alpha)),beta);
            J31=zeros(sizh(2),length(beta));%jacobianestim(@(x)sum(eelog(pop,x,alpha)./repmat(denom,[1,sizs(2)])),beta);
   
            J11=[J11;J21([1:lb,(lb+la+1):end],:)-[zeros(lb,lb);J21((lb+1):(lb+la),:)];J31];
            
            J41=jacobianestim(@(x)sum(ufunc_log(pop,beta,x,theta)),alpha);
            J51=jacobianestim(@(x)sum(eelog(pop,beta,x)),alpha);
            J61=zeros(sizh(2),length(alpha));
            
            J21=[J41;J51([1:lb,(lb+la+1):end],:)-[zeros(lb,la);J51((lb+1):(lb+la),:)];J61];
            
            %J=[J1,J2];
            J71=jacobianestim(@(x)sum(ufunc_log(pop,beta,alpha,x)),theta);
            J81=zeros(sizs(2),length(theta));
            J91=jacobianestim(@(x)sum(hfunc_log(pop(:,1),pop(:,3),x)),theta);
            
            J31=[J71;J81([1:lb,(lb+la+1):end],:)-J81([1:lb,(lb+1):(lb+la)],:);J91];
            J1=[J11,J21,J31];
     v2=diag(inv(J1'*inv(u'*u)*J1));
     v2=sqrt(v2(1:lb));
            