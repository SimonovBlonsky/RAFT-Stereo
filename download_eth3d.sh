mkdir datasets/ETH3D/two_view_testing -p
cd datasets/ETH3D/two_view_testing
wget https://www.eth3d.net/data/two_view_test.7z
echo "Unzipping two_view_test.7z using p7zip (installed from environment.yaml)"
7za x two_view_test.7z
cd ../../..