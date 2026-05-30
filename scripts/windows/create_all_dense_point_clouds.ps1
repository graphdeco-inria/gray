$ErrorActionPreference = "Stop"

$scenes = @(
    @{ Path = '360_v2/bicycle'; Resize = '4'; RomaModel = 'outdoors' }
    @{ Path = '360_v2/flowers'; Resize = '4'; RomaModel = 'outdoors' }
    @{ Path = '360_v2/garden'; Resize = '4'; RomaModel = 'outdoors' }
    @{ Path = '360_v2/treehill'; Resize = '4'; RomaModel = 'outdoors' }
    @{ Path = '360_v2/stump'; Resize = '4'; RomaModel = 'outdoors' }
    @{ Path = '360_v2/bonsai'; Resize = '2'; RomaModel = 'indoors' }
    @{ Path = '360_v2/counter'; Resize = '2'; RomaModel = 'indoors' }
    @{ Path = '360_v2/kitchen'; Resize = '2'; RomaModel = 'indoors' }
    @{ Path = '360_v2/room'; Resize = '2'; RomaModel = 'indoors' }
    @{ Path = 'tandt/train'; Resize = '1'; RomaModel = 'outdoors' }
    @{ Path = 'tandt/truck'; Resize = '1'; RomaModel = 'outdoors' }
    @{ Path = 'db/drjohnson'; Resize = '1'; RomaModel = 'indoors' }
    @{ Path = 'db/playroom'; Resize = '1'; RomaModel = 'indoors' }
)

$index = 0
$total = $scenes.Count

foreach ($scene in $scenes) {
    $index += 1
    Write-Host "Creating dense point cloud for scene $($scene.Path) [$index/$total]"
    python third_party/edgs.py -s "data/$($scene.Path)" -r $scene.Resize --roma_model $scene.RomaModel
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
