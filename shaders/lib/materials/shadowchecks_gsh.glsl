if (mat0 < 1000) {
    tracemat = false;
    if (abs(cnormal.y) > 0.99 && area > 0.85) {
        float blockCenterDist = abs(fract(avgPos.y) - 0.5);
        if (blockCenterDist > 0.4) mat0 = 1008;
        else if (cnormal.y > 0) mat0 = 1009;
        else mat0 = 1010;
        zpos = 0.52 - blockCenterDist;
        tracemat = true;
    } else if (abs(cnormal.y) < 0.01) {
        if (max(abs(cnormal.x), abs(cnormal.z)) > 0.99 && area > 0.85) {
            mat0 = 1008;
            zpos = -0.999;
            tracemat = true;
        }
        else if (abs(abs(cnormal.x) - abs(cnormal.z)) < 0.01) {
            mat0 = 1004;
            tracemat = true;
        }
    }
}
switch(mat0) {
    case 50004:
        coord = vec2(0.5 / shadowMapResolution);
        zpos = -avgPos.y / (VXHEIGHT * VXHEIGHT);
        break;
    default:
        coord = getVxCoords(avgPos);
        if (coord.x < 1.0 / shadowMapResolution) return;
}
switch (mat0) {
    case 10064:
        if (dot(cnormal, vec3(0, 1, 0)) < 0.95) tracemat = false;
        else avgPos -= 0.05 * cnormal;
        break;

    case 31000:
        zpos = 0.3 * zpos + 0.7;
    case 10068:
        if (area < 0.8) tracemat = false;
    case 10584:
    case 10588:
        doCuboidTexCoordCorrection = false;
        break;
    case 10350:
        if (cnormal.y < 0.5) tracemat = false;
        avgPos.y -= 0.1;
        break;
    case 12380:
        //tracemat = false;
        break;
    case 10548:
        if(area < 0.8) tracemat = false;
        break;
    case 10496:
    case 10528:
    case 10604:
    case 12604:
        if (cnormal.y < 0.5) tracemat = false;
        //avgPos += vec3(0.0, 0.1, 0.0);
        break;
    case 10544:
    case 10596:
    case 10600:
    case 12112:
    case 12173:
        //avgPos += 0.1 * cnormal;
        break;
    case 60008:
    case 60012:
        if (area < 0.3) tracemat = false;
        break;
    case 0:
    case 10472:
    case 50016:
    case 50996:
    case 60004: 
    case 60018:
        tracemat = false;
        break;
    default:
        avgPos -= 0.04 * cnormal;
        break;
}
if (mat0 >= 60000) {
    if (cnormal.y < 0.5) tracemat = false;
    avgPos.y -= 0.3;
}