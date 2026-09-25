function B = dct2(A)
  [m,n] = size(A);
  Tm = zeros(m); Tn = zeros(n);
  for k = 0:m-1
    for j = 0:m-1
      Tm(k+1,j+1) = sqrt(2/m)*cos(pi*(2*j+1)*k/(2*m));
    end
  end
  for k = 0:n-1
    for j = 0:n-1
      Tn(k+1,j+1) = sqrt(2/n)*cos(pi*(2*j+1)*k/(2*n));
    end
  end
  Tm(1,:) = Tm(1,:)/sqrt(2);
  Tn(1,:) = Tn(1,:)/sqrt(2);
  B = Tm*A*Tn';
end
