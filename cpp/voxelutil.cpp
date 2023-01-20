#include"pybind11/pybind11.h"
#include"pybind11/numpy.h"
#include<vector>
#include<algorithm>
#include<unordered_map>

namespace py = pybind11;
using pyint = py::ssize_t;
using py::array_t;
using std::vector;
using std::unordered_map;
using std::swap;
using std::reverse;

const float eps = 1e-6;
const int maxn = 51;

int sig(float d) {
    return(d > eps) - (d < -eps);
}
struct Point {
    float x, y; Point() {}
    Point(float x, float y) :x(x), y(y) {}
    bool operator==(const Point& p)const {
        return sig(x - p.x) == 0 && sig(y - p.y) == 0;
    }
};
float cross(Point o, Point a, Point b) {
    return(a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}
float area(Point* ps, int n) {
    ps[n] = ps[0];
    float res = 0;
    for (int i = 0; i < n; i++) {
        res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
    }
    return res / 2.0;
}
int lineCross(Point a, Point b, Point c, Point d, Point& p) {
    float s1, s2;
    s1 = cross(a, b, c);
    s2 = cross(a, b, d);
    if (sig(s1) == 0 && sig(s2) == 0) return 2;
    if (sig(s2 - s1) == 0) return 0;
    p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
    p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
    return 1;
}

void polygon_cut(Point* p, int& n, Point a, Point b) {
    static Point pp[20];
    int m = 0; p[n] = p[0];
    for (int i = 0; i < n; i++) {
        if (sig(cross(a, b, p[i])) > 0) pp[m++] = p[i];
        if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1])))
            lineCross(a, b, p[i], p[i + 1], pp[m++]);
    }
    n = 0;
    for (int i = 0; i < m; i++)
        if (!i || !(pp[i] == pp[i - 1]))
            p[n++] = pp[i];
    while (n > 1 && p[n - 1] == p[0])n--;
}

float intersectArea(Point a, Point b, Point c, Point d) {
    Point o(0, 0);
    int s1 = sig(cross(o, a, b));
    int s2 = sig(cross(o, c, d));
    if (s1 == 0 || s2 == 0)return 0.0;
    if (s1 == -1) swap(a, b);
    if (s2 == -1) swap(c, d);
    Point p[10] = { o,a,b };
    int n = 3;
    polygon_cut(p, n, o, c);
    polygon_cut(p, n, c, d);
    polygon_cut(p, n, d, o);
    float res = fabs(area(p, n));
    if (s1 * s2 == -1) res = -res; return res;
}

float intersectArea(Point* ps1, int n1, Point* ps2, int n2) {
    if (area(ps1, n1) < 0) reverse(ps1, ps1 + n1);
    if (area(ps2, n2) < 0) reverse(ps2, ps2 + n2);
    ps1[n1] = ps1[0];
    ps2[n2] = ps2[0];
    float res = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
        }
    }
    return res;
}

auto classifyAnchors(array_t<float> gts, array_t<float> anchors, array_t<long long> nls, array_t<long long> nws,
    float negThr, float posThr) {
    static Point r1[5], r2[5];
    int anchorsPerLoc = anchors.shape(2);
    auto _gts = gts.unchecked<3>();
    auto _anchors = anchors.unchecked<5>();
    auto _nls = nls.unchecked<1>();
    auto _nws = nws.unchecked<1>();
    vector<vector<pyint>> pi(3), ni(3);
    vector<pyint> gi;
    for (int j = 0; j < 4; j++) {
        r2[j].x = _anchors(0, 0, 0, j, 0), r2[j].y = _anchors(0, 0, 0, j, 1);
    }
    float anchorArea = area(r2, 4);
    pyint l = anchors.shape(0), w = anchors.shape(1);
    for (pyint i = 0; i < gts.shape(0); i++) {
        pyint nl = _nls(i), nw = _nws(i);
        for (int j = 0; j < 4; j++) {
            r1[j].x = _gts(i, j, 0), r1[j].y = _gts(i, j, 1);
        }
        float gtArea = area(r1, 4);
        for (pyint z = 0; z < anchorsPerLoc; z++) {
            for (pyint h = 0; nl + h < l; h++) {
                for (int j = 0; j < 4; j++) {
                    r2[j].x = _anchors(nl + h, nw, z, j, 0), r2[j].y = _anchors(nl + h, nw, z, j, 1);
                }
                float inter = intersectArea(r1, 4, r2, 4);
                float iou = inter / (gtArea + anchorArea - inter);
                if (iou < 0.1) {
                    break;
                }
                if (iou >= posThr) {
                    /*_pos(nl + h, nw, z) = true;
                    _gi(nl + h, nw, z) = i;
                    _neg(nl + h, nw, z) = false;*/
                    pi[0].push_back(nl + h);
                    pi[1].push_back(nw);
                    pi[2].push_back(z);
                    gi.push_back(i);
                    ni[0].push_back(nl + h);
                    ni[1].push_back(nw);
                    ni[2].push_back(z);
                } else if (iou >= negThr) {
                    //_neg(nl + h, nw, z) = false;
                    ni[0].push_back(nl + h);
                    ni[1].push_back(nw);
                    ni[2].push_back(z);
                }
                for (pyint v = 1; nw + v < w; v++) {
                    for (int j = 0; j < 4; j++) {
                        r2[j].x = _anchors(nl + h, nw + v, z, j, 0), r2[j].y = _anchors(nl + h, nw + v, z, j, 1);
                    }
                    inter = intersectArea(r1, 4, r2, 4);
                    iou = inter / (gtArea + anchorArea - inter);
                    if (iou < 0.1) {
                        break;
                    }
                    if (iou >= posThr) {
                        /*_pos(nl + h, nw + v, z) = true;
                        _gi(nl + h, nw + v, z) = i;
                        _neg(nl + h, nw + v, z) = false;*/
                        pi[0].push_back(nl + h);
                        pi[1].push_back(nw + v);
                        pi[2].push_back(z);
                        gi.push_back(i);
                        ni[0].push_back(nl + h);
                        ni[1].push_back(nw + v);
                        ni[2].push_back(z);
                    } else if (iou >= negThr) {
                        //_neg(nl + h, nw + v, z) = false;
                        ni[0].push_back(nl + h);
                        ni[1].push_back(nw + v);
                        ni[2].push_back(z);
                    }
                }
                for (pyint v = -1; nw + v >= 0; v--) {
                    for (int j = 0; j < 4; j++) {
                        r2[j].x = _anchors(nl + h, nw + v, z, j, 0), r2[j].y = _anchors(nl + h, nw + v, z, j, 1);
                    }
                    inter = intersectArea(r1, 4, r2, 4);
                    iou = inter / (gtArea + anchorArea - inter);
                    if (iou < 0.1) {
                        break;
                    }
                    if (iou >= posThr) {
                        pi[0].push_back(nl + h);
                        pi[1].push_back(nw + v);
                        pi[2].push_back(z);
                        gi.push_back(i);
                        ni[0].push_back(nl + h);
                        ni[1].push_back(nw + v);
                        ni[2].push_back(z);
                    } else if (iou >= negThr) {
                        ni[0].push_back(nl + h);
                        ni[1].push_back(nw + v);
                        ni[2].push_back(z);
                    }
                }
            }
            for (pyint h = -1; nl + h >= 0; h--) {
                for (int j = 0; j < 4; j++) {
                    r2[j].x = _anchors(nl + h, nw, z, j, 0), r2[j].y = _anchors(nl + h, nw, z, j, 1);
                }
                float inter = intersectArea(r1, 4, r2, 4);
                float iou = inter / (gtArea + anchorArea - inter);
                if (iou < 0.1) {
                    break;
                }
                if (iou >= posThr) {
                    /*_pos(nl + h, nw, z) = true;
                    _gi(nl + h, nw, z) = i;
                    _neg(nl + h, nw, z) = false;*/
                    pi[0].push_back(nl + h);
                    pi[1].push_back(nw);
                    pi[2].push_back(z);
                    gi.push_back(i);
                    ni[0].push_back(nl + h);
                    ni[1].push_back(nw);
                    ni[2].push_back(z);
                } else if (iou >= negThr) {
                    //_neg(nl + h, nw, z) = false;
                    ni[0].push_back(nl + h);
                    ni[1].push_back(nw);
                    ni[2].push_back(z);
                }
                for (pyint v = 1; nw + v < w; v++) {
                    for (int j = 0; j < 4; j++) {
                        r2[j].x = _anchors(nl + h, nw + v, z, j, 0), r2[j].y = _anchors(nl + h, nw + v, z, j, 1);
                    }
                    inter = intersectArea(r1, 4, r2, 4);
                    iou = inter / (gtArea + anchorArea - inter);
                    if (iou < 0.1) {
                        break;
                    }
                    if (iou >= posThr) {
                        pi[0].push_back(nl + h);
                        pi[1].push_back(nw + v);
                        pi[2].push_back(z);
                        gi.push_back(i);
                        ni[0].push_back(nl + h);
                        ni[1].push_back(nw + v);
                        ni[2].push_back(z);
                    } else if (iou >= negThr) {
                        ni[0].push_back(nl + h);
                        ni[1].push_back(nw + v);
                        ni[2].push_back(z);
                    }
                }
                for (pyint v = -1; nw + v >= 0; v--) {
                    for (int j = 0; j < 4; j++) {
                        r2[j].x = _anchors(nl + h, nw + v, z, j, 0), r2[j].y = _anchors(nl + h, nw + v, z, j, 1);
                    }
                    inter = intersectArea(r1, 4, r2, 4);
                    iou = inter / (gtArea + anchorArea - inter);
                    if (iou < 0.1) {
                        break;
                    }
                    if (iou >= posThr) {
                        pi[0].push_back(nl + h);
                        pi[1].push_back(nw + v);
                        pi[2].push_back(z);
                        gi.push_back(i);
                        ni[0].push_back(nl + h);
                        ni[1].push_back(nw + v);
                        ni[2].push_back(z);
                    } else if (iou >= negThr) {
                        ni[0].push_back(nl + h);
                        ni[1].push_back(nw + v);
                        ni[2].push_back(z);
                    }
                }
            }
        }
    }
    array_t<pyint> px(gi.size(), pi[0].data()), py(gi.size(), pi[1].data()), pz(gi.size(), pi[2].data());
    array_t<pyint> nx(ni[0].size(), ni[0].data()), ny(ni[1].size(), ni[1].data()), nz(ni[2].size(), ni[2].data());
    array_t<pyint> gires(gi.size(), gi.data());
    auto pires = py::make_tuple(px, py, pz), nires = py::make_tuple(nx, ny, nz);
    return py::make_tuple(pires, nires, gires);
}

std::hash<int> intHash;
struct tripleHash {
    int operator()(const std::tuple<int, int, int> t) const {
        return intHash(std::get<0>(t)) ^ intHash(std::get<1>(t)) ^ intHash(std::get<2>(t));
    }
};

auto group(array_t<float> pcd, array_t<int> idx, int samplesPerVoxel) {
    unordered_map<std::tuple<int, int, int>, int, tripleHash> mp;
    auto _pcd = pcd.unchecked<2>();
    auto _idx = idx.unchecked<2>();
    vector<vector<pyint>> uidx(3);
    vector<vector<pyint>> voxel;
    for (pyint i = 0; i < idx.shape(0); i++) {
        auto it = mp.find({ _idx(i, 0), _idx(i, 1), _idx(i, 2) });
        if (it == mp.end()) {
            mp[{_idx(i, 0), _idx(i, 1), _idx(i, 2)}] = voxel.size();
            for (int j = 0; j < 3; j++) {
                uidx[j].push_back(_idx(i, j));
            }
            voxel.push_back({i});
        } else if (voxel[it->second].size() < samplesPerVoxel) {
            voxel[it->second].push_back(i);
        }
    }

    array_t<pyint> cnt(voxel.size());
    array_t<float> voxelres({ (int)voxel.size(), samplesPerVoxel, 7 });
    voxelres[py::ellipsis()] = 0;
    auto _cnt = cnt.mutable_unchecked<1>();
    auto _voxelres = voxelres.mutable_unchecked<3>();
    for (size_t i = 0; i < voxel.size(); i++) {
        _cnt(i) = voxel[i].size();
        for (size_t j = 0; j < voxel[i].size(); j++) {
            for (int k = 0; k < 3; k++) {
                _voxelres(i, j, k) = _pcd(voxel[i][j], k);
            }
            _voxelres(i, j, 6) = _pcd(voxel[i][j], 3);
        }
    }
    array_t<pyint> x(voxel.size(), uidx[0].data()), y(voxel.size(), uidx[1].data()), z(voxel.size(), uidx[2].data());
    return py::make_tuple(voxelres, py::make_tuple(x, y, z), cnt);
}

PYBIND11_MODULE(voxelutil, m) {
    m.doc() = "Some utility functions for VoxelNet";
    m.def("_classifyAnchors", &classifyAnchors);
    m.def("_group", &group);
}